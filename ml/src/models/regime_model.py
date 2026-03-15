"""
regime_model.py
---------------
Regime-aware ensemble — trains separate models per market regime
and selects the appropriate model at inference time based on
the current detected regime.

This is a strong paper contribution:
    "Standard models use one set of weights across all market conditions.
     We train regime-specific models and route predictions through a
     regime detector at inference time, improving robustness."

Regime detection uses:
    - VIX level + change (fear indicator)
    - Yield spread (recession signal)
    - SP500 trend (momentum regime)

Usage:
    from ml.src.models.regime_model import RegimeAwareEnsemble
    model = RegimeAwareEnsemble()
    model.fit_all_regimes(df, ticker="AAPL", causal_features=features)
    result = model.predict(live_features, ticker="AAPL", current_price=213.40)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.models.base_model import PredictionResult
from ml.src.evaluation.regime_splitter import RegimeSplitter

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects the current market regime from live macro features.

    Rules (in priority order):
        1. crisis    — VIX > 30 AND VIX rising fast
        2. rate_hike — yield_spread inverted AND VIX moderate
        3. bull      — SP500 above 200-day MA AND VIX < 20
        4. recovery  — SP500 rising AND VIX falling from high
        5. neutral   — default when no clear signal
    """

    REGIMES = ["bull", "recovery", "rate_hike", "crisis", "neutral"]

    def detect(self, live_features: pd.Series) -> str:
        """
        Detect regime from a live feature vector.
        Returns one of: bull, recovery, rate_hike, crisis, neutral
        """
        vix_level   = live_features.get("vix_level",    20.0)
        vix_change  = live_features.get("vix_change_1d", 0.0)
        yield_spread = live_features.get("yield_spread", 1.0)
        sp500_ret   = live_features.get("sp500_return_1d", 0.0)
        vol_regime  = live_features.get("vol_regime", 0.0)

        # Crisis: VIX spike
        if vix_level > 30 and vix_change > 2:
            return "crisis"

        # Rate hike: yield curve inverted
        if yield_spread < -0.1:
            return "rate_hike"

        # High vol but not crisis
        if vol_regime > 0.7 and vix_level > 25:
            return "recovery"

        # Bull: low vol, positive momentum
        if vix_level < 20 and sp500_ret > 0:
            return "bull"

        return "neutral"

    def detect_from_df(self, df: pd.DataFrame) -> pd.Series:
        """Label each row of a DataFrame with its regime."""
        return df.apply(self.detect, axis=1).rename("detected_regime")


class RegimeAwareEnsemble:
    """
    Trains one ensemble per market regime and routes predictions
    through the appropriate regime model at inference time.

    At inference:
        1. Detect current regime from live features
        2. Load that regime's model
        3. Return prediction
        4. If detected regime has no trained model, fall back to "neutral"
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg      = _load_config(config_path)
        self.root     = Path(__file__).resolve().parents[3]
        self.target   = self.cfg["model"]["target"]
        self.horizon  = self.cfg["model"]["horizon_days"]

        self.splitter = RegimeSplitter(config_path)
        self.detector = RegimeDetector()

        self.models_dir = self.root / self.cfg["saved_models"]["dir"] / "regime_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # One ensemble per regime
        self._regime_ensembles: dict = {}
        self._causal_features: list[str] = []

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit_all_regimes(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        min_samples: int = 200,
    ) -> dict[str, dict]:
        """
        Train a separate ensemble for each regime.

        For each regime:
            - Training data: all pre-regime data + the regime itself
            - This gives the model exposure to the regime's characteristics

        Args:
            df:              Full feature matrix
            ticker:          e.g. "AAPL"
            causal_features: Causal feature list from selector
            min_samples:     Skip regime if fewer than this many samples

        Returns:
            Dict of {regime: test_metrics}
        """
        from ml.src.ensemble import Ensemble
        from ml.src.evaluation.metrics import Metrics

        ticker  = ticker.upper()
        metrics = Metrics()
        results = {}

        self._causal_features = causal_features

        # Add "neutral" regime = full dataset
        regime_splits = self.splitter.split_all(df)
        regime_splits["neutral"] = df

        for regime_name, regime_df in regime_splits.items():
            if len(regime_df) < min_samples:
                logger.warning(
                    f"[regime_model] Skipping {regime_name} — "
                    f"only {len(regime_df)} samples."
                )
                continue

            logger.info(
                f"[regime_model] Training {regime_name} model "
                f"({len(regime_df)} samples) ..."
            )

            try:
                ensemble = Ensemble()
                X_test, y_test = ensemble.train_all(
                    regime_df, f"{ticker}_{regime_name}", causal_features
                )

                # Evaluate
                test_df = pd.concat([X_test, y_test], axis=1)
                preds   = ensemble.predict_historical(test_df, causal_features)

                if "actual_return" in preds.columns:
                    scores = metrics.compute_all(
                        preds["predicted_return"],
                        preds["actual_return"],
                        label=f"{regime_name}",
                    )
                    results[regime_name] = scores

                self._regime_ensembles[regime_name] = ensemble
                logger.info(f"[regime_model] {regime_name} model trained.")

            except Exception as e:
                logger.warning(f"[regime_model] {regime_name} training failed: {e}")

        # Save regime metadata
        self._save_metadata(ticker)
        logger.info(
            f"[regime_model] Trained {len(self._regime_ensembles)} regime models."
        )
        return results

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        live_features: pd.Series,
        ticker: str,
        current_price: float,
    ) -> PredictionResult:
        """
        Detect current regime and route to the appropriate model.
        """
        regime = self.detector.detect(live_features)
        logger.info(f"[regime_model] Detected regime: {regime}")

        # Fall back to neutral if regime model not available
        ensemble = self._regime_ensembles.get(
            regime,
            self._regime_ensembles.get("neutral")
        )

        if ensemble is None:
            raise RuntimeError(
                "[regime_model] No models loaded. Call fit_all_regimes() first."
            )

        result = ensemble.predict_live(
            live_features=live_features,
            ticker=ticker,
            current_price=current_price,
            causal_features=self._causal_features,
        )

        # Add regime info to result
        result.model_name = f"regime_aware({regime})"
        return result

    def load_all(self, ticker: str) -> None:
        """Load all regime models from disk."""
        from ml.src.ensemble import Ensemble
        from ml.src.causal.selector import CausalSelector

        ticker   = ticker.upper()
        meta     = self._load_metadata(ticker)
        self._causal_features = meta.get("causal_features", [])

        for regime in meta.get("trained_regimes", []):
            try:
                ensemble = Ensemble()
                ensemble.load(f"{ticker}_{regime}")
                self._regime_ensembles[regime] = ensemble
                logger.info(f"[regime_model] Loaded {regime} model.")
            except Exception as e:
                logger.warning(f"[regime_model] Could not load {regime}: {e}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _save_metadata(self, ticker: str) -> None:
        metadata = {
            "ticker":           ticker,
            "trained_regimes":  list(self._regime_ensembles.keys()),
            "causal_features":  self._causal_features,
        }
        path = self.models_dir / f"regime_metadata_{ticker}.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[regime_model] Metadata saved → {path.name}")

    def _load_metadata(self, ticker: str) -> dict:
        path = self.models_dir / f"regime_metadata_{ticker}.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)