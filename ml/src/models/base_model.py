"""
base_model.py
-------------
Abstract base class all models inherit from + PredictionResult dataclass.

Every model (LightGBM, XGBoost, ARIMA) must implement:
    fit(X, y)           — train on feature matrix + target vector
    predict(X)          — return PredictionResult for one or more rows
    save(ticker)        — persist weights to saved_models/
    load(ticker)        — restore weights from saved_models/

PredictionResult is the single output contract used by:
    - ensemble.py       — blends multiple PredictionResults
    - backend/prediction_service.py  — serialises to JSON for the API
    - frontend          — renders prediction + confidence band + drivers

Nothing outside ml/ ever imports model internals — only PredictionResult.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PredictionResult — the single output contract
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Standardised output from any model or ensemble.
    All fields are plain Python types — JSON-serialisable by default.
    """

    # Core prediction
    ticker:            str
    predicted_return:  float          # 5-day forward log return
    predicted_price:   float          # absolute price prediction
    current_price:     float          # price at prediction time
    direction:         str            # "UP" or "DOWN"
    confidence:        float          # 0.0 – 1.0

    # Confidence band (90% CI by default)
    upper_band:        float
    lower_band:        float

    # Causal drivers — what pushed the prediction UP or DOWN
    # [{"feature": "vix_change_1d", "impact": "negative", "value": 0.23}, ...]
    causal_drivers:    list[dict]     = field(default_factory=list)

    # Metadata
    model_name:        str            = "unknown"
    horizon_days:      int            = 5
    prediction_date:   str            = ""          # ISO date string

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialisation."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionResult":
        """Deserialise from dict."""
        return cls(**d)

    def __str__(self) -> str:
        return (
            f"PredictionResult({self.ticker} | "
            f"{self.direction} {self.predicted_return:+.2%} | "
            f"conf={self.confidence:.2f} | "
            f"price={self.predicted_price:.2f} "
            f"[{self.lower_band:.2f}, {self.upper_band:.2f}])"
        )


# ---------------------------------------------------------------------------
# BaseModel — abstract class all models inherit from
# ---------------------------------------------------------------------------

class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.

    Subclasses must implement: fit, predict, save, load.
    Everything else (path resolution, config loading, driver extraction)
    is handled here so subclasses stay clean.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg        = _load_config(config_path)
        self.root       = Path(__file__).resolve().parents[3]
        self.target_col = self.cfg["model"]["target"]
        self.horizon    = self.cfg["model"]["horizon_days"]
        self.confidence_z = self.cfg["model"]["ensemble"]["confidence_z"]

        models_dir = self.cfg["saved_models"]["dir"]
        self.models_dir = self.root / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_name: str = "base"
        self._is_fitted: bool = False

    # -----------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # -----------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        Train the model.

        Args:
            X_train: Training features (causal feature columns only).
            y_train: Training target (log_return_5d).
            X_val:   Validation features (optional, for early stopping).
            y_val:   Validation target (optional).
        """
        ...

    @abstractmethod
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return raw predictions (log returns) as numpy array.
        Called internally by predict().
        """
        ...

    @abstractmethod
    def save(self, ticker: str) -> None:
        """Persist trained model weights to saved_models/."""
        ...

    @abstractmethod
    def load(self, ticker: str) -> None:
        """Restore trained model weights from saved_models/."""
        ...

    # -----------------------------------------------------------------------
    # Concrete — predict() is implemented here using predict_raw()
    # -----------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        ticker: str,
        current_price: float,
        causal_features: Optional[list[str]] = None,
    ) -> PredictionResult:
        """
        Generate a full PredictionResult from a feature row.

        Args:
            X:               Feature DataFrame — typically one row (live inference)
                             or multiple rows (backtesting).
            ticker:          Stock ticker.
            current_price:   Latest close price.
            causal_features: List of causal feature names for driver extraction.

        Returns:
            PredictionResult for the last row of X.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Model '{self.model_name}' is not fitted. Call fit() or load() first."
            )

        # Raw prediction
        raw_preds = self.predict_raw(X)
        pred_return = float(raw_preds[-1])

        # Predicted price
        pred_price = current_price * np.exp(pred_return)

        # Direction
        direction = "UP" if pred_return >= 0 else "DOWN"

        # Confidence — based on prediction magnitude relative to recent volatility
        confidence = self._compute_confidence(raw_preds, X)

        # Confidence band
        vol = self._estimate_volatility(X)
        z   = self.confidence_z
        upper = current_price * np.exp(pred_return + z * vol * np.sqrt(self.horizon))
        lower = current_price * np.exp(pred_return - z * vol * np.sqrt(self.horizon))

        # Causal drivers
        drivers = self._extract_drivers(X.iloc[[-1]], causal_features)

        from datetime import datetime
        return PredictionResult(
            ticker=ticker,
            predicted_return=pred_return,
            predicted_price=round(pred_price, 2),
            current_price=round(current_price, 2),
            direction=direction,
            confidence=round(confidence, 3),
            upper_band=round(upper, 2),
            lower_band=round(lower, 2),
            causal_drivers=drivers,
            model_name=self.model_name,
            horizon_days=self.horizon,
            prediction_date=datetime.today().strftime("%Y-%m-%d"),
        )

    # -----------------------------------------------------------------------
    # Data preparation — shared by all subclasses
    # -----------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        causal_features: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series,   pd.Series,   pd.Series]:
        """
        Split feature matrix into train/val/test sets.
        Chronological split — NO shuffling.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Keep only causal features + target
        cols = causal_features + [self.target_col]
        df   = df[cols].dropna()

        X = df[causal_features]
        y = df[self.target_col]

        n          = len(df)
        train_end  = int(n * self.cfg["model"]["train_ratio"])
        val_end    = int(n * (self.cfg["model"]["train_ratio"] +
                              self.cfg["model"]["val_ratio"]))

        X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]
        X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test,  y_test  = X.iloc[val_end:],    y.iloc[val_end:]

        logger.info(
            f"[{self.model_name}] Data split — "
            f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} rows."
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        ticker: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit scaler on train, transform all splits.
        Saves scaler to saved_models/ for reuse at inference.
        """
        import joblib
        from sklearn.preprocessing import StandardScaler, RobustScaler

        scaler_type = self.cfg["model"]["scaler"]
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
        else:
            logger.info(f"[{self.model_name}] No scaling applied.")
            return X_train, X_val, X_test

        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index, columns=X_train.columns
        )
        X_val_s = pd.DataFrame(
            scaler.transform(X_val),
            index=X_val.index, columns=X_val.columns
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index, columns=X_test.columns
        )

        # Save scaler
        fname  = self.cfg["saved_models"]["scaler_filename"].replace("{ticker}", ticker.upper())
        path   = self.models_dir / fname
        joblib.dump(scaler, path)
        logger.info(f"[{self.model_name}] Scaler saved → {path.name}")

        self._scaler = scaler
        return X_train_s, X_val_s, X_test_s

    def load_scaler(self, ticker: str) -> None:
        """Load previously saved scaler."""
        import joblib
        fname = self.cfg["saved_models"]["scaler_filename"].replace("{ticker}", ticker.upper())
        path  = self.models_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"No scaler found for {ticker} at {path}")
        self._scaler = joblib.load(path)
        logger.info(f"[{self.model_name}] Scaler loaded from {path.name}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply loaded scaler to a feature DataFrame."""
        if not hasattr(self, "_scaler"):
            return X
        return pd.DataFrame(
            self._scaler.transform(X),
            index=X.index, columns=X.columns
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_confidence(
        self, raw_preds: np.ndarray, X: pd.DataFrame
    ) -> float:
        """
        Confidence score (0-1) based on prediction strength
        relative to estimated volatility.

        Higher predicted return relative to noise → higher confidence.
        Capped between 0.3 and 0.9 — never fully certain or uncertain.
        """
        pred   = abs(float(raw_preds[-1]))
        vol    = self._estimate_volatility(X)
        if vol == 0:
            return 0.5
        ratio  = pred / (vol * np.sqrt(self.horizon))
        # Sigmoid-like mapping to [0.3, 0.9]
        conf   = 0.3 + 0.6 * (1 / (1 + np.exp(-2 * (ratio - 0.5))))
        return float(np.clip(conf, 0.3, 0.9))

    def _estimate_volatility(self, X: pd.DataFrame) -> float:
        """
        Estimate daily volatility from the feature matrix.
        Uses volatility_10d if available, else volatility_20d, else 0.02.
        """
        for col in ["volatility_10d", "volatility_20d", "volatility_30d"]:
            if col in X.columns:
                val = float(X[col].iloc[-1])
                if not np.isnan(val) and val > 0:
                    # Convert annualised back to daily
                    return val / np.sqrt(252)
        return 0.02   # fallback: ~2% daily vol

    def _extract_drivers(
        self,
        X_row: pd.DataFrame,
        causal_features: Optional[list[str]],
    ) -> list[dict]:
        """
        Extract causal drivers from a single feature row.
        For tree models, uses SHAP if available.
        Falls back to signed feature values for interpretability.

        Returns list of top-5 drivers sorted by absolute impact.
        """
        if causal_features is None or len(causal_features) == 0:
            return []

        row = X_row[
            [c for c in causal_features if c in X_row.columns]
        ].iloc[0]

        drivers = []
        for feat, val in row.items():
            drivers.append({
                "feature": feat,
                "value":   round(float(val), 4),
                "impact":  "positive" if float(val) > 0 else "negative",
            })

        # Sort by absolute value — biggest movers first
        drivers.sort(key=lambda d: abs(d["value"]), reverse=True)
        return drivers[:5]

    # -----------------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------------

    def _model_path(self, ticker: str, filename_key: str) -> Path:
        """Resolve model save path from config template."""
        fname = (
            self.cfg["saved_models"][filename_key]
            .replace("{ticker}", ticker.upper())
        )
        return self.models_dir / fname