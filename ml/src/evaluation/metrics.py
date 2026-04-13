"""
metrics.py  (CORRECTED)
-----------------------
Replaces the original metrics.py.

Key fixes:
    1. sharpe_ratio(): adds turnover-aware transaction costs.
       Original applied 10bps on EVERY period regardless of whether
       the model changed direction. This overstates costs when the model
       holds the same direction for multiple periods in a row.
       Fix: only charge 10bps when direction changes (actual turnover).

    2. sharpe_ratio_scaled(): new method using confidence-weighted positions.
       Original used binary ±1 positions. Real strategies scale position
       size by predicted return magnitude or confidence. This method
       shows what a practitioner would actually achieve.

    3. r2_score(): added explicit note about expected range for log returns
       (0.01-0.10 is good — don't compare to price-level R²).

    4. mae() and r2_score() are always included in compute_all() output
       (required for the paper's regression evaluation methodology).

All original methods preserved with same signatures — backward compatible.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class Metrics:
    """
    All evaluation metrics used in the paper.
    Backward compatible with the original — all method signatures unchanged.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg     = _load_config(config_path)
        trading = cfg["evaluation"]["trading"]

        self.risk_free_rate  = trading["risk_free_rate_annual"]
        self.trading_days    = trading["trading_days_per_year"]
        self.tx_cost_bps     = trading["transaction_cost_bps"]
        self.initial_capital = trading["initial_capital"]
        self.metrics_list    = cfg["evaluation"]["metrics"]

    # -----------------------------------------------------------------------
    # compute_all
    # -----------------------------------------------------------------------

    def compute_all(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
        label: str = "",
    ) -> dict[str, float]:
        """Compute all configured metrics. Backward compatible."""
        y_pred, y_true = self._align(y_pred, y_true)
        scores = {}

        if "directional_accuracy" in self.metrics_list:
            scores["directional_accuracy"] = self.directional_accuracy(y_pred, y_true)

        if "sharpe_ratio" in self.metrics_list:
            # Use turnover-aware Sharpe (corrected version)
            scores["sharpe_ratio"] = self.sharpe_ratio(y_pred, y_true)

        if "rmse" in self.metrics_list:
            scores["rmse"] = self.rmse(y_pred, y_true)

        # Always include mae and r2 for regression paper evaluation
        scores["mae"]      = self.mae(y_pred, y_true)
        scores["r2_score"] = self.r2_score(y_pred, y_true)

        if "mape" in self.metrics_list:
            scores["mape"] = self.mape(y_pred, y_true)

        if "max_drawdown" in self.metrics_list:
            scores["max_drawdown"] = self.max_drawdown(y_pred, y_true)

        if "calmar_ratio" in self.metrics_list:
            scores["calmar_ratio"] = self.calmar_ratio(y_pred, y_true)

        prefix = f"[{label}] " if label else ""
        logger.info(
            f"{prefix}Metrics — "
            + " | ".join(f"{k}={v:.4f}" for k, v in scores.items())
        )
        return scores

    # -----------------------------------------------------------------------
    # directional_accuracy
    # -----------------------------------------------------------------------

    def directional_accuracy(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """Percentage of correct UP/DOWN direction calls. Random = 0.50."""
        y_pred, y_true = self._align(y_pred, y_true)
        correct = ((y_pred >= 0) == (y_true >= 0)).sum()
        return float(correct / len(y_true))

    # -----------------------------------------------------------------------
    # sharpe_ratio — CORRECTED with turnover-aware costs
    # -----------------------------------------------------------------------

    def sharpe_ratio(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> float:
        """
        Annualised Sharpe ratio of a long/short strategy.

        FIX vs original: transaction costs are only charged when the
        model CHANGES direction (actual turnover), not on every period.

        Original bug: applied 10bps every 5 trading days regardless
        of direction change. This was equivalent to 600bps/year in
        costs even if the model held the same direction continuously.
        That overstated the impact of transaction costs.

        Corrected: charge 10bps only on periods where sign(pred[t]) ≠ sign(pred[t-1]).
        This better reflects the actual cost of a signal-following strategy.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true

        # ── CORRECTED: turnover-aware transaction costs ────────────────────
        # Turnover occurs when direction changes: sign flips from +1 to -1 or vice versa
        direction_changes  = np.diff(positions, prepend=positions[0])
        turned_over        = np.abs(direction_changes) > 0
        tx_cost_per_period = (self.tx_cost_bps / 10_000) * turned_over.astype(float)
        strat_rets_net     = strat_rets - tx_cost_per_period

        # Original used a flat cost every period — kept here as a comment
        # for comparison in the paper:
        # strat_rets_net = strat_rets - (self.tx_cost_bps / 10_000)

        daily_rf = self.risk_free_rate / self.trading_days
        excess   = strat_rets_net - daily_rf
        std      = strat_rets_net.std()

        if std == 0 or np.isnan(std):
            return 0.0

        sharpe = float((excess.mean() / std) * np.sqrt(self.trading_days))

        # Log turnover stats for transparency
        n_trades   = int(turned_over.sum())
        turnover_pct = n_trades / len(positions) * 100
        logger.debug(
            f"[metrics] Sharpe={sharpe:.4f}, turnover={n_trades}/{len(positions)} "
            f"periods ({turnover_pct:.1f}%)"
        )
        return sharpe

    def sharpe_ratio_original(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> float:
        """
        Original Sharpe ratio with flat 10bps cost every period.
        Kept for comparison in the paper — shows original vs corrected.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true - (self.tx_cost_bps / 10_000)
        daily_rf   = self.risk_free_rate / self.trading_days
        excess     = strat_rets - daily_rf
        std        = strat_rets.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * np.sqrt(self.trading_days))

    def sharpe_ratio_scaled(
        self,
        y_pred:      pd.Series,
        y_true:      pd.Series,
        confidence:  Optional[pd.Series] = None,
    ) -> float:
        """
        Confidence-weighted Sharpe ratio.

        Instead of binary ±1 positions, scales position size by the
        model's calibrated confidence (or by |predicted return| if
        confidence is not provided).

        This reflects real-world usage: a practitioner would size
        positions larger when the model is more confident.

        Args:
            y_pred:     Predicted returns
            y_true:     Actual returns
            confidence: Calibrated confidence scores (0-1).
                        If None, uses |y_pred| normalised to [0, 1].

        Returns:
            Annualised Sharpe ratio with scaled positions.
        """
        y_pred, y_true = self._align(y_pred, y_true)

        if confidence is not None:
            conf, _ = self._align(confidence, y_true)
            scale   = conf.values
        else:
            # Use normalised |predicted return| as position scale
            abs_pred = np.abs(y_pred.values)
            max_pred = abs_pred.max()
            scale    = abs_pred / max_pred if max_pred > 0 else np.ones_like(abs_pred)

        positions  = np.sign(y_pred.values) * scale
        strat_rets = positions * y_true.values

        # Turnover-aware costs
        prev_pos  = np.roll(positions, 1)
        prev_pos[0] = 0.0
        turnover  = np.abs(positions - prev_pos)
        strat_rets_net = strat_rets - (self.tx_cost_bps / 10_000) * turnover

        daily_rf = self.risk_free_rate / self.trading_days
        excess   = strat_rets_net - daily_rf
        std      = strat_rets_net.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * np.sqrt(self.trading_days))

    # -----------------------------------------------------------------------
    # rmse, mae, r2_score, mape — unchanged from original
    # -----------------------------------------------------------------------

    def rmse(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        y_pred, y_true = self._align(y_pred, y_true)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def mae(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        y_pred, y_true = self._align(y_pred, y_true)
        return float(np.mean(np.abs(y_pred - y_true)))

    def r2_score(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        """
        R² on log returns. Expected range: 0.01-0.10 is GOOD for financial
        return prediction. Do NOT compare to price-level R² (~0.99).
        """
        y_pred, y_true = self._align(y_pred, y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)

    def mape(
        self, y_pred: pd.Series, y_true: pd.Series,
        epsilon: float = 1e-8,
    ) -> float:
        y_pred, y_true = self._align(y_pred, y_true)
        denom = np.abs(y_true) + epsilon
        return float(np.mean(np.abs(y_pred - y_true) / denom))

    def max_drawdown(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        y_pred, y_true = self._align(y_pred, y_true)
        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true
        equity     = (1 + strat_rets).cumprod()
        peak       = equity.cummax()
        dd         = (equity - peak) / peak
        return float(dd.min())

    def calmar_ratio(self, y_pred: pd.Series, y_true: pd.Series) -> float:
        y_pred, y_true = self._align(y_pred, y_true)
        positions      = np.where(y_pred >= 0, 1, -1)
        strat_rets     = positions * y_true
        annual_return  = strat_rets.mean() * self.trading_days
        mdd            = abs(self.max_drawdown(y_pred, y_true))
        if mdd == 0:
            return 0.0
        return float(annual_return / mdd)

    # -----------------------------------------------------------------------
    # Comparison table and baseline random
    # -----------------------------------------------------------------------

    def comparison_table(
        self, results: dict[str, dict[str, float]]
    ) -> pd.DataFrame:
        """Build paper comparison table from multiple model result dicts."""
        df       = pd.DataFrame(results).T
        df.index.name = "model"
        pct_cols   = ["directional_accuracy", "max_drawdown"]
        float_cols = ["sharpe_ratio", "calmar_ratio", "rmse", "mae", "r2_score", "mape"]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.1%}")
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.4f}")
        return df

    def baseline_random(self, y_true: pd.Series, n_trials: int = 1000) -> dict:
        """Expected metrics for a random direction predictor (bootstrap)."""
        rng    = np.random.default_rng(42)
        scores = []
        for _ in range(n_trials):
            y_random = pd.Series(
                rng.choice([-0.01, 0.01], size=len(y_true)),
                index=y_true.index,
            )
            scores.append(self.compute_all(y_random, y_true))
        result = {}
        for k in scores[0].keys():
            vals = [s[k] for s in scores]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        return result

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _align(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        combined = pd.concat([y_pred, y_true], axis=1).dropna()
        return combined.iloc[:, 0], combined.iloc[:, 1]