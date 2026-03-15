"""
metrics.py
----------
All evaluation metrics used in the paper.

Metrics computed:
    directional_accuracy   — % of correct UP/DOWN calls
    sharpe_ratio           — annualised risk-adjusted return
    rmse                   — root mean squared error on log returns
    mape                   — mean absolute percentage error on log returns
    max_drawdown           — worst peak-to-trough equity curve drop
    calmar_ratio           — annualised return / max drawdown

All metrics are computed from:
    y_pred: pd.Series of predicted log returns
    y_true: pd.Series of actual log returns

Usage:
    from ml.src.evaluation.metrics import Metrics
    m = Metrics()

    scores = m.compute_all(y_pred, y_true)
    print(scores)

    # Individual metrics
    da  = m.directional_accuracy(y_pred, y_true)
    sr  = m.sharpe_ratio(y_pred)
    mdd = m.max_drawdown(y_pred)
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

        Args:
            y_pred: Predicted log returns (aligned index with y_true)
            y_true: Actual log returns
            label:  Optional label for logging (e.g. regime name)

        Returns:
            Dict: metric_name → float value
        """
        y_pred, y_true = self._align(y_pred, y_true)

        scores = {}

        if "directional_accuracy" in self.metrics_list:
            scores["directional_accuracy"] = self.directional_accuracy(y_pred, y_true)

        if "sharpe_ratio" in self.metrics_list:
            scores["sharpe_ratio"] = self.sharpe_ratio(y_pred, y_true)

        if "rmse" in self.metrics_list:
            scores["rmse"] = self.rmse(y_pred, y_true)

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
        Random baseline = 0.50.
        Anything above 0.55 consistently is meaningful.

        Returns: float in [0, 1]
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

        Strategy:
            - Go LONG  (return = actual return)  when prediction is UP
            - Go SHORT (return = -actual return) when prediction is DOWN
            - Apply transaction cost on every trade

        Sharpe = (mean_daily_return - daily_rf) / std_daily_return * sqrt(252)

        Returns: float (can be negative)
        """
        y_pred, y_true = self._align(y_pred, y_true)

        # Strategy returns
        positions   = np.where(y_pred >= 0, 1, -1)
        strat_rets  = positions * y_true

        # Apply transaction cost (bps converted to decimal)
        tx_cost     = self.tx_cost_bps / 10_000
        # Cost on every day (simple approximation — cost each prediction period)
        strat_rets  = strat_rets - tx_cost

        daily_rf    = self.risk_free_rate / self.trading_days
        excess_rets = strat_rets - daily_rf
        std         = strat_rets.std()

        if std == 0 or np.isnan(std):
            return 0.0

        sharpe = (excess_rets.mean() / std) * np.sqrt(self.trading_days)
        return float(sharpe)

    def rmse(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> float:
        """
        Root Mean Squared Error on log returns.
        Lower is better. Heavily penalises large errors.

        Returns: float >= 0
        """
        y_pred, y_true = self._align(y_pred, y_true)
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def mape(
        self, y_pred: pd.Series, y_true: pd.Series,
        epsilon: float = 1e-8,
    ) -> float:
        """
        Mean Absolute Percentage Error on log returns.
        Uses epsilon to avoid division by zero on near-zero returns.

        Returns: float >= 0 (as a ratio, not %)
        """
        y_pred, y_true = self._align(y_pred, y_true)
        denom = np.abs(y_true) + epsilon
        return float(np.mean(np.abs(y_pred - y_true) / denom))

    def max_drawdown(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> float:
        """
        Maximum drawdown of the long/short strategy equity curve.
        Measures worst peak-to-trough drop in cumulative returns.

        Returns: float in [-1, 0] (negative — drawdown is a loss)
        """
        y_pred, y_true = self._align(y_pred, y_true)

        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true

        # Cumulative equity curve
        equity = (1 + strat_rets).cumprod()
        peak   = equity.cummax()
        dd     = (equity - peak) / peak

        return float(dd.min())

    def calmar_ratio(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
    ) -> float:
        """
        Calmar ratio = annualised return / abs(max drawdown).
        Measures return per unit of drawdown risk.
        Higher is better.

        Returns: float (can be negative if strategy loses money)
        """
        y_pred, y_true = self._align(y_pred, y_true)

        positions  = np.where(y_pred >= 0, 1, -1)
        strat_rets = positions * y_true

        # Annualised return
        mean_daily    = strat_rets.mean()
        annual_return = mean_daily * self.trading_days

        mdd = abs(self.max_drawdown(y_pred, y_true))
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

        Args:
            results: {model_or_regime_name: {metric: value}}
                     e.g. {
                         "all_features":  {"directional_accuracy": 0.54, ...},
                         "granger":       {"directional_accuracy": 0.55, ...},
                         "pcmci_causal":  {"directional_accuracy": 0.57, ...},
                     }

        Returns:
            DataFrame with models as rows, metrics as columns.
        """
        df = pd.DataFrame(results).T
        df.index.name = "model"

        # Format percentages for readability
        pct_cols = ["directional_accuracy", "max_drawdown"]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.1%}")

        float_cols = ["sharpe_ratio", "calmar_ratio", "rmse", "mape"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.3f}")

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

        Returns: {metric: mean_value, metric_std: std_value}
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
        keys   = scores[0].keys()
        for k in keys:
            vals = [s[k] for s in scores]
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
        return result