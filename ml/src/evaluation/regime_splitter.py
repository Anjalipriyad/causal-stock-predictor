"""
regime_splitter.py
------------------
Labels the feature matrix by market regime and splits it for regime-specific
evaluation. Regime definitions are loaded from config.yaml.

Regimes:
    bull:        2010-01-01 → 2019-12-31   pre-COVID bull run
    covid_crash: 2020-01-01 → 2020-06-30   crash + initial shock
    recovery:    2020-07-01 → 2021-12-31   stimulus-driven recovery
    rate_hike:   2022-01-01 → 2022-12-31   Fed tightening cycle
    ai_bull:     2023-01-01 → 2025-12-31   AI-driven bull run

Usage:
    from ml.src.evaluation.regime_splitter import RegimeSplitter
    splitter = RegimeSplitter()

    # Get data for one regime
    df_crash = splitter.get_regime(df, "covid_crash")

    # Get all regimes as a dict
    regimes = splitter.split_all(df)

    # Add regime label column to df
    df_labelled = splitter.label(df)
"""

import logging
from typing import Optional

import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class RegimeSplitter:
    """
    Splits a feature matrix DataFrame into named market regime windows.
    All date ranges are read from config.yaml — nothing hardcoded.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)

        # Parse regime date ranges from config
        raw_regimes = cfg["evaluation"]["regimes"]
        self.regimes: dict[str, tuple[str, str]] = {}
        for name, dates in raw_regimes.items():
            self.regimes[name] = (dates[0], dates[1])

        logger.info(
            f"[regime_splitter] Loaded {len(self.regimes)} regimes: "
            f"{list(self.regimes.keys())}"
        )

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def get_regime(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """
        Return subset of df matching a named regime's date range.

        Args:
            df:     Feature matrix with DatetimeIndex
            regime: Regime name e.g. "covid_crash"

        Returns:
            DataFrame subset for that regime period.
        """
        if regime not in self.regimes:
            raise ValueError(
                f"Unknown regime '{regime}'. "
                f"Available: {list(self.regimes.keys())}"
            )
        start, end = self.regimes[regime]
        subset = df.loc[start:end]

        if subset.empty:
            logger.warning(
                f"[regime_splitter] No data for regime '{regime}' "
                f"({start} → {end})."
            )
        else:
            logger.info(
                f"[regime_splitter] Regime '{regime}': "
                f"{len(subset)} rows ({start} → {end})"
            )
        return subset

    def split_all(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Split df into all configured regimes.

        Returns:
            Dict: regime_name → DataFrame subset
        """
        result = {}
        for name in self.regimes:
            subset = self.get_regime(df, name)
            if not subset.empty:
                result[name] = subset
        return result

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'regime' string column to df labelling each row's market regime.
        Rows that fall outside all defined regime windows are labelled 'other'.

        Args:
            df: Feature matrix with DatetimeIndex

        Returns:
            df copy with added 'regime' column
        """
        df = df.copy()
        df["regime"] = "other"

        for name, (start, end) in self.regimes.items():
            mask = (df.index >= start) & (df.index <= end)
            df.loc[mask, "regime"] = name

        counts = df["regime"].value_counts()
        logger.info(f"[regime_splitter] Regime label counts:\n{counts}")
        return df

    def regime_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics per regime.
        Useful for the paper — shows how each regime differs.

        Returns:
            DataFrame with regime as index and stats as columns:
                n_rows, date_start, date_end,
                mean_return, std_return, min_return, max_return
        """
        target_col = None
        for col in ["log_return_5d", "log_return_1d", "actual_return"]:
            if col in df.columns:
                target_col = col
                break

        rows = []
        for name, subset in self.split_all(df).items():
            row = {
                "regime":     name,
                "n_rows":     len(subset),
                "date_start": str(subset.index.min().date()),
                "date_end":   str(subset.index.max().date()),
            }
            if target_col:
                row["mean_return"] = round(float(subset[target_col].mean()), 4)
                row["std_return"]  = round(float(subset[target_col].std()),  4)
                row["min_return"]  = round(float(subset[target_col].min()),  4)
                row["max_return"]  = round(float(subset[target_col].max()),  4)
            rows.append(row)

        return pd.DataFrame(rows).set_index("regime")

    def train_test_split_by_regime(
        self,
        df: pd.DataFrame,
        test_regime: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split df into train (all regimes except test_regime) and
        test (test_regime only).

        Useful for out-of-distribution evaluation — train on everything
        before a specific regime, test on that regime.

        Args:
            df:           Full feature matrix
            test_regime:  Regime to hold out as test set

        Returns:
            train_df, test_df
        """
        test_df  = self.get_regime(df, test_regime)
        start, _ = self.regimes[test_regime]
        train_df = df.loc[:start].iloc[:-1]   # everything before test regime

        logger.info(
            f"[regime_splitter] Train/test split by regime '{test_regime}': "
            f"train={len(train_df)}, test={len(test_df)}"
        )
        return train_df, test_df

    @property
    def regime_names(self) -> list[str]:
        """Return list of all configured regime names."""
        return list(self.regimes.keys())

    @property
    def regime_dates(self) -> dict[str, tuple[str, str]]:
        """Return dict of regime_name → (start_date, end_date)."""
        return dict(self.regimes)