"""
calibration.py
--------------
Probability calibration for the ensemble's confidence scores.

Problem: the original _compute_confidence() in base_model.py maps
|pred| / (vol * sqrt(horizon)) through a sigmoid to [0.3, 0.9].
This is a heuristic — a confidence of 0.7 does NOT mean the model
is right 70% of the time.

Fix: use isotonic regression to map raw confidence scores to
empirically calibrated probabilities. After calibration, a
confidence of 0.7 means the model's directional calls are correct
~70% of the time when it reports that confidence level.

Paper section: Section 4.4 — "Confidence Calibration"
    "We calibrate the ensemble's directional confidence scores using
     isotonic regression fitted on held-out validation set predictions.
     A reliability diagram (Figure X) shows that the calibrated
     probabilities are well-matched to empirical accuracy across
     confidence bins."

Usage:
    from ml.src.models.calibration import ConfidenceCalibrator

    # After training ensemble:
    calibrator = ConfidenceCalibrator()
    calibrator.fit(val_raw_confidences, val_correct_directions)
    calibrator.save(ticker, models_dir)

    # At inference:
    calibrator.load(ticker, models_dir)
    calibrated_conf = calibrator.transform(raw_confidence)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Isotonic regression calibrator for directional confidence scores.

    Isotonic regression is preferred over Platt scaling (logistic)
    because:
    1. Financial confidence distributions are not sigmoid-shaped
    2. Isotonic is non-parametric — makes no distributional assumptions
    3. It strictly monotone-increases the mapping, preserving ranking

    Limitation: requires sufficient calibration samples (>200 recommended).
    With fewer samples, use Platt scaling (set method='platt').
    """

    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'isotonic' (default) or 'platt' (logistic regression)
        """
        assert method in ("isotonic", "platt"), \
            f"method must be 'isotonic' or 'platt', got '{method}'"
        self.method      = method
        self._calibrator = None
        self._is_fitted  = False
        self._n_samples  = 0
        self._mean_acc   = None

    def fit(
        self,
        raw_confidences: np.ndarray,
        correct_directions: np.ndarray,
    ) -> "ConfidenceCalibrator":
        """
        Fit the calibrator on held-out validation data.

        MUST be fitted on validation data only, never training data.
        Training data confidence scores are overfit — using them would
        produce a miscalibrated map.

        Args:
            raw_confidences:    Array of raw confidence scores from ensemble
                                Shape: (n_samples,), values in [0, 1]
            correct_directions: Binary array: 1 if direction correct, 0 if wrong
                                Shape: (n_samples,)

        Returns:
            self (fitted)
        """
        raw_confidences    = np.array(raw_confidences).ravel()
        correct_directions = np.array(correct_directions).ravel().astype(float)

        assert len(raw_confidences) == len(correct_directions)
        n = len(raw_confidences)
        self._n_samples = n
        self._mean_acc  = float(correct_directions.mean())

        if n < 50:
            logger.warning(
                f"[calibration] Only {n} samples for calibration. "
                f"Results may be unreliable (need ≥ 200 ideally)."
            )

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._calibrator = IsotonicRegression(
                out_of_bounds="clip",
                increasing=True,
            )
            self._calibrator.fit(raw_confidences, correct_directions)

        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self._calibrator = LogisticRegression(solver="lbfgs")
            self._calibrator.fit(raw_confidences.reshape(-1, 1), correct_directions)

        self._is_fitted = True

        # Log calibration quality (Expected Calibration Error)
        ece = self._compute_ece(raw_confidences, correct_directions)
        calibrated = self.transform(raw_confidences)
        ece_after  = self._compute_ece(calibrated, correct_directions)

        logger.info(
            f"[calibration] Fitted on {n} samples. "
            f"ECE before={ece:.4f}, after={ece_after:.4f}. "
            f"Mean accuracy={self._mean_acc:.3f}"
        )
        return self

    def transform(self, raw_confidences: np.ndarray) -> np.ndarray:
        """
        Map raw confidence scores to calibrated probabilities.

        Args:
            raw_confidences: Array of raw scores, shape (n,)

        Returns:
            Calibrated probabilities, shape (n,), values in [0, 1]
        """
        if not self._is_fitted:
            logger.warning(
                "[calibration] Not fitted — returning raw confidences unchanged."
            )
            return np.array(raw_confidences)

        raw = np.array(raw_confidences).ravel()

        if self.method == "isotonic":
            calibrated = self._calibrator.predict(raw)
        elif self.method == "platt":
            calibrated = self._calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

        return np.clip(calibrated, 0.0, 1.0)

    def transform_scalar(self, raw_confidence: float) -> float:
        """Convenience wrapper for single confidence score."""
        return float(self.transform(np.array([raw_confidence]))[0])

    # -----------------------------------------------------------------------
    # Reliability diagram data (for paper figure)
    # -----------------------------------------------------------------------

    def reliability_data(
        self,
        raw_confidences: np.ndarray,
        correct_directions: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute reliability diagram data for the paper.

        A well-calibrated model's reliability diagram lies on the diagonal:
        when the model says 0.7 confidence, ~70% of those calls are correct.

        Args:
            raw_confidences:    Raw or calibrated confidence scores
            correct_directions: Binary correct/wrong array
            n_bins:             Number of bins for the diagram

        Returns:
            DataFrame with columns:
                bin_center:       midpoint of confidence bin
                mean_confidence:  average confidence in this bin
                fraction_correct: empirical accuracy in this bin
                count:            number of predictions in this bin
        """
        raw  = np.array(raw_confidences).ravel()
        corr = np.array(correct_directions).ravel().astype(float)

        bins  = np.linspace(0, 1, n_bins + 1)
        rows  = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask   = (raw >= lo) & (raw < hi)
            if mask.sum() == 0:
                continue
            rows.append({
                "bin_low":          round(lo, 3),
                "bin_high":         round(hi, 3),
                "bin_center":       round((lo + hi) / 2, 3),
                "mean_confidence":  float(raw[mask].mean()),
                "fraction_correct": float(corr[mask].mean()),
                "count":            int(mask.sum()),
            })

        return pd.DataFrame(rows)

    def print_reliability_table(
        self,
        raw_confidences:    np.ndarray,
        correct_directions: np.ndarray,
        n_bins: int = 10,
    ) -> None:
        """Print before/after calibration reliability table for the paper."""
        raw  = np.array(raw_confidences)
        cal  = self.transform(raw) if self._is_fitted else raw
        corr = np.array(correct_directions).astype(float)

        raw_data = self.reliability_data(raw, corr, n_bins)
        cal_data = self.reliability_data(cal, corr, n_bins)

        ece_before = self._compute_ece(raw, corr)
        ece_after  = self._compute_ece(cal, corr) if self._is_fitted else ece_before

        print(f"\n{'='*65}")
        print(f"  RELIABILITY DIAGRAM DATA")
        print(f"  ECE before calibration: {ece_before:.4f}")
        print(f"  ECE after  calibration: {ece_after:.4f}")
        print(f"  (lower ECE = better calibration | 0.0 = perfect)")
        print(f"{'='*65}")
        print(f"  {'Bin':>8}  {'Raw conf':>10}  {'Frac correct':>13}  "
              f"{'Cal conf':>10}  {'Cal correct':>12}  {'Count':>6}")
        print(f"  {'-'*65}")
        for _, rrow in raw_data.iterrows():
            crow = cal_data[cal_data["bin_center"] == rrow["bin_center"]]
            cal_conf_str    = f"{crow['mean_confidence'].iloc[0]:.3f}" if not crow.empty else "N/A"
            cal_correct_str = f"{crow['fraction_correct'].iloc[0]:.3f}" if not crow.empty else "N/A"
            print(
                f"  [{rrow['bin_low']:.2f},{rrow['bin_high']:.2f}]  "
                f"{rrow['mean_confidence']:>10.3f}  "
                f"{rrow['fraction_correct']:>13.3f}  "
                f"{cal_conf_str:>10}  {cal_correct_str:>12}  "
                f"{rrow['count']:>6}"
            )
        print(f"{'='*65}\n")

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, ticker: str, models_dir: Path) -> None:
        """Save calibrator to saved_models/."""
        if not self._is_fitted:
            logger.warning("[calibration] Not fitted — nothing to save.")
            return
        path = models_dir / f"calibrator_{ticker.upper()}.pkl"
        joblib.dump({
            "calibrator":  self._calibrator,
            "method":      self.method,
            "n_samples":   self._n_samples,
            "mean_acc":    self._mean_acc,
        }, path)
        logger.info(f"[calibration] Saved → {path.name}")

    def load(self, ticker: str, models_dir: Path) -> None:
        """Load calibrator from saved_models/."""
        path = models_dir / f"calibrator_{ticker.upper()}.pkl"
        if not path.exists():
            logger.warning(
                f"[calibration] No calibrator found for {ticker} at {path}. "
                "Confidence scores will be uncalibrated."
            )
            return
        data = joblib.load(path)
        self._calibrator = data["calibrator"]
        self.method      = data["method"]
        self._n_samples  = data.get("n_samples", 0)
        self._mean_acc   = data.get("mean_acc", None)
        self._is_fitted  = True
        logger.info(
            f"[calibration] Loaded from {path.name} "
            f"(method={self.method}, n_samples={self._n_samples})"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_ece(
        self,
        confidences: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error.
        ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
        Lower is better. Perfect calibration = 0.0.
        """
        bins  = np.linspace(0, 1, n_bins + 1)
        n     = len(confidences)
        ece   = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask   = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            acc  = correct[mask].mean()
            conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)