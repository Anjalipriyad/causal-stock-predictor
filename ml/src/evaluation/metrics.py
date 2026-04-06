"""
metrics.py
----------
All evaluation metrics used in the paper.

Metrics computed:
    directional_accuracy   — % of correct UP/DOWN calls
    sharpe_ratio           — annualised risk-adjusted return
    rmse                   — root mean squared error on log returns
    mae                    — mean absolute error on log returns
    r2_score               — coefficient of determination on log returns
    mape                   — mean absolute percentage error on log returns
    max_drawdown           — worst peak-to-trough equity curve drop
    calmar_ratio           — annualised return / max drawdown

mae and r2_score are added to match the regression paper's evaluation
methodology. Note: R² on log returns will typically be low (0.01-0.10 is
considered good for financial return prediction) — this is normal and
expected. Do NOT compare to R² scores on raw price levels, which are
artificially inflated by price level trends.

Usage:
    from ml.src.evaluation.metrics import Metrics
    m = Metrics()

    scores = m.compute_all(y_pred, y_true)
    print(scores)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class Metrics:
    """
    Computes all evaluation metrics from predicted and actual return series.
    All config-driven — thresholds and constants read from config.yaml.
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
    # Public — compute all at once
    # -----------------------------------------------------------------------

    def compute_all(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
        label: str = "",
    ) -> dict[str, float]:
        """
        Compute all configured metrics and return as a dict.
        Also always includes mae and r2_score regardless of config,
        as these are required for the regression paper evaluation.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        scores = {}

        if "directional_accuracy" in self.metrics_list:
            scores["directional_accuracy"] = self.directional_accuracy(y_pred, y_true)

        if "sharpe_ratio" in self.metrics_list:
            scores["sharpe_ratio"] = self.sharpe_ratio(y_pred, y_true)

        if "rmse" in self.metrics_list:
            scores["rmse"] = self.rmse(y_pred, y_true)

        # MAE and R² always included (required for regression paper)
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
    # Individual metrics
    # -----------------------------------------------------------------------

    def directional_accuracy(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """
        Percentage of predictions with correct UP/DOWN direction.
        Random baseline = 0.50. Anything above 0.55 consistently is meaningful.
        Returns float in [0, 1].
        """
        y_pred, y_true = self._align(y_pred, y_true)
        correct = ((y_pred >= 0) == (y_true >= 0)).sum()
        return float(correct / len(y_true))

    def sharpe_ratio(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> float:
        """
        Annualised Sharpe ratio of a simple long/short strategy.
        Strategy: LONG when prediction is UP, SHORT when DOWN.
        Transaction cost applied on every prediction period.
        Returns float (can be negative).
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

    def rmse(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """
        Root Mean Squared Error on log returns.
        Lower is better. Heavily penalises large errors.
        Returns float >= 0.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def mae(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """
        Mean Absolute Error on log returns.
        Average magnitude of prediction error.
        Less sensitive to large outliers than RMSE.
        Returns float >= 0.

        Interpretation for Nifty:
            If log return is ~0.01 (1%) daily, MAE of 0.005 means the model
            is off by about 0.5% on average — roughly half a percent per day.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        return float(np.mean(np.abs(y_pred - y_true)))

    def r2_score(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """
        R² (coefficient of determination) on log returns.

        Values:
            1.0  = perfect prediction
            0.0  = model is no better than predicting the mean
            <0.0 = model is worse than predicting the mean

        IMPORTANT: R² on LOG RETURNS is expected to be low (0.01-0.10 is good).
        This is normal for financial return prediction — returns have very low
        signal-to-noise. Do NOT compare this R² to values from price-level
        regression (which are artificially inflated by price trends and would
        be ~0.99 for any naive model).
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
        """
        Mean Absolute Percentage Error on log returns.
        Uses epsilon to avoid division by zero on near-zero returns.
        Returns float >= 0 (as a ratio, not %).
        """
        y_pred, y_true = self._align(y_pred, y_true)
        denom = np.abs(y_true) + epsilon
        return float(np.mean(np.abs(y_pred - y_true) / denom))

    def max_drawdown(
        self, y_pred: pd.Series, y_true: pd.Series,
    ) -> float:
        """
        Maximum drawdown of the long/short strategy equity curve.
        Returns float in [-1, 0] (negative — drawdown is a loss).
        """
        y_pred, y_true = self._align(y_pred, y_true)
        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true
        equity     = (1 + strat_rets).cumprod()
        peak       = equity.cummax()
        dd         = (equity - peak) / peak
        return float(dd.min())

    def calmar_ratio(
        self, y_pred: pd.Series, y_true: pd.Series,
    ) -> float:
        """
        Calmar ratio = annualised return / abs(max drawdown).
        Higher is better. Returns float (can be negative).
        """
        y_pred, y_true = self._align(y_pred, y_true)
        positions      = np.where(y_pred >= 0, 1, -1)
        strat_rets     = positions * y_true
        annual_return  = strat_rets.mean() * self.trading_days
        mdd            = abs(self.max_drawdown(y_pred, y_true))
        if mdd == 0:
            return 0.0
        return float(annual_return / mdd)

    # -----------------------------------------------------------------------
    # Comparison table — for paper
    # -----------------------------------------------------------------------

    def comparison_table(
        self,
        results: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """
        Build a comparison table of metrics across multiple models/regimes.
        This is Table 2 in the paper.
        """
        df = pd.DataFrame(results).T
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

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _align(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Align pred and true on common index, drop NaN rows."""
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        combined = pd.concat([y_pred, y_true], axis=1).dropna()
        return combined.iloc[:, 0], combined.iloc[:, 1]

    def baseline_random(self, y_true: pd.Series, n_trials: int = 1000) -> dict:
        """
        Compute expected metrics for a random direction predictor.
        Useful for establishing statistical significance in the paper.
        """
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