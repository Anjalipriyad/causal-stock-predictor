"""
validator.py
------------
Data quality checks run after loading, before feature engineering.
Catches problems early so they don't silently corrupt the model.

Three levels of checks:
    1. Structure   — correct columns, correct index type
    2. Completeness — NaN counts, gap detection, stale data
    3. Sanity      — price > 0, volume > 0, no future dates

Usage:
    from ml.src.data.validator import DataValidator
    validator = DataValidator()

    # Raises or warns depending on severity
    validator.validate_prices(df, ticker="AAPL")
    validator.validate_macro(df, symbol="^VIX")
    validator.validate_sentiment(df, ticker="AAPL")

    # Full report without raising
    report = validator.report(df, name="AAPL prices")
    print(report)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    name:     str
    passed:   bool                  = True
    errors:   list[str]             = field(default_factory=list)
    warnings: list[str]             = field(default_factory=list)
    stats:    dict                  = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False
        logger.error(f"[validator] {self.name}: {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"[validator] {self.name}: {msg}")

    def __str__(self) -> str:
        lines = [
            f"=== Validation Report: {self.name} ===",
            f"Status : {'PASSED' if self.passed else 'FAILED'}",
            f"Errors : {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            lines.extend(f"  ✗ {e}" for e in self.errors)
        if self.warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        if self.stats:
            lines.append("\nStats:")
            lines.extend(f"  {k}: {v}" for k, v in self.stats.items())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates DataFrames produced by DataLoader before they enter
    the feature engineering pipeline.
    """

    # Expected columns per data type
    PRICE_COLUMNS     = {"open", "high", "low", "close", "volume"}
    SENTIMENT_COLUMNS = {"article_count", "avg_sentiment"}
    MACRO_COLUMNS     = {"close"}

    # Thresholds
    MAX_NAN_RATIO         = 0.05    # >5% NaN → error
    MAX_GAP_DAYS          = 10      # gap > 10 trading days → warning
    MAX_STALE_DAYS        = 30      # last date > 30 days ago → warning
    MIN_ROWS              = 200     # fewer rows than this → error
    MAX_PRICE_CHANGE_PCT  = 0.50    # single-day change > 50% → warning (split?)

    def __init__(self, strict: bool = True):
        """
        strict=True  → raise ValueError on errors
        strict=False → log errors but don't raise (useful in notebooks)
        """
        self.strict = strict

    # -----------------------------------------------------------------------
    # Public validators
    # -----------------------------------------------------------------------

    def validate_prices(
        self, df: pd.DataFrame, ticker: str = "unknown"
    ) -> ValidationReport:
        """Full validation for OHLCV price DataFrame."""
        report = ValidationReport(name=f"{ticker} prices")
        self._check_not_empty(df, report)
        if not report.passed:
            return self._finalise(report)

        self._check_index(df, report)
        self._check_columns(df, self.PRICE_COLUMNS, report, required=True)
        self._check_nan_ratio(df, report)
        self._check_min_rows(df, report)
        self._check_date_gaps(df, report)
        self._check_stale(df, report)
        self._check_price_positive(df, report)
        self._check_price_jumps(df, report)
        self._check_volume_positive(df, report)
        self._add_stats(df, report)

        return self._finalise(report)

    def validate_macro(
        self, df: pd.DataFrame, symbol: str = "unknown"
    ) -> ValidationReport:
        """Validation for macro / sector ETF DataFrame."""
        report = ValidationReport(name=f"{symbol} macro")
        self._check_not_empty(df, report)
        if not report.passed:
            return self._finalise(report)

        self._check_index(df, report)
        self._check_columns(df, self.MACRO_COLUMNS, report, required=False)
        self._check_nan_ratio(df, report)
        self._check_min_rows(df, report)
        self._check_date_gaps(df, report)
        self._check_stale(df, report)
        self._add_stats(df, report)

        return self._finalise(report)

    def validate_sentiment(
        self, df: pd.DataFrame, ticker: str = "unknown"
    ) -> ValidationReport:
        """Validation for sentiment DataFrame."""
        report = ValidationReport(name=f"{ticker} sentiment")
        self._check_not_empty(df, report)
        if not report.passed:
            return self._finalise(report)

        self._check_index(df, report)
        self._check_columns(df, self.SENTIMENT_COLUMNS, report, required=True)
        self._check_nan_ratio(df, report)
        self._check_sentiment_range(df, report)
        self._check_stale(df, report)
        self._add_stats(df, report)

        return self._finalise(report)

    def validate_feature_matrix(
        self, df: pd.DataFrame, name: str = "features"
    ) -> ValidationReport:
        """
        Validation for the final merged feature matrix produced by pipeline.py.
        More strict — this is what goes into the model.
        """
        report = ValidationReport(name=name)
        self._check_not_empty(df, report)
        if not report.passed:
            return self._finalise(report)

        self._check_index(df, report)
        self._check_nan_ratio(df, report, max_ratio=0.01)   # stricter: 1%
        self._check_min_rows(df, report)
        self._check_no_infinite(df, report)
        self._check_constant_columns(df, report)
        self._add_stats(df, report)

        return self._finalise(report)

    def report(self, df: pd.DataFrame, name: str = "data") -> ValidationReport:
        """Generic report without type-specific checks. Useful in notebooks."""
        report = ValidationReport(name=name)
        self._check_not_empty(df, report)
        if report.passed:
            self._check_index(df, report)
            self._check_nan_ratio(df, report)
            self._check_min_rows(df, report)
            self._check_no_infinite(df, report)
            self._add_stats(df, report)
        return self._finalise(report)

    # -----------------------------------------------------------------------
    # Private checks
    # -----------------------------------------------------------------------

    def _check_not_empty(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        if df is None or df.empty:
            report.add_error("DataFrame is empty.")

    def _check_index(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            report.add_error(
                f"Index must be DatetimeIndex, got {type(df.index).__name__}."
            )

    def _check_columns(
        self,
        df: pd.DataFrame,
        expected: set[str],
        report: ValidationReport,
        required: bool = True,
    ) -> None:
        missing = expected - set(df.columns)
        if missing:
            msg = f"Missing columns: {sorted(missing)}"
            if required:
                report.add_error(msg)
            else:
                report.add_warning(msg)

    def _check_nan_ratio(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        max_ratio: Optional[float] = None,
    ) -> None:
        max_ratio = max_ratio or self.MAX_NAN_RATIO
        nan_ratio = df.isnull().mean()
        bad_cols  = nan_ratio[nan_ratio > max_ratio]
        if not bad_cols.empty:
            for col, ratio in bad_cols.items():
                report.add_error(
                    f"Column '{col}' has {ratio:.1%} NaN values (max {max_ratio:.0%})."
                )
        # Also warn on any NaN
        total_nan = df.isnull().sum().sum()
        if total_nan > 0:
            report.add_warning(f"Total NaN cells: {total_nan}")

    def _check_min_rows(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        if len(df) < self.MIN_ROWS:
            report.add_error(
                f"Only {len(df)} rows — need at least {self.MIN_ROWS}."
            )

    def _check_date_gaps(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Detect gaps larger than MAX_GAP_DAYS in the date index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        diffs = df.index.to_series().diff().dropna()
        max_gap = diffs.max()
        if pd.notna(max_gap) and max_gap.days > self.MAX_GAP_DAYS:
            # Find where the gap occurs
            gap_loc = diffs.idxmax()
            report.add_warning(
                f"Largest date gap: {max_gap.days} days at {gap_loc.date()}."
            )

    def _check_stale(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Warn if the most recent date is more than MAX_STALE_DAYS ago."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        last_date  = df.index.max()
        days_since = (datetime.today() - last_date).days
        if days_since > self.MAX_STALE_DAYS:
            report.add_warning(
                f"Most recent data is {days_since} days old ({last_date.date()})."
                " Consider re-running load_historical()."
            )

    def _check_price_positive(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                n_bad = (df[col] <= 0).sum()
                if n_bad > 0:
                    report.add_error(
                        f"Column '{col}' has {n_bad} non-positive values."
                    )

    def _check_price_jumps(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Warn on single-day price changes > 50% — may indicate bad data or splits."""
        if "close" not in df.columns:
            return
        pct_change = df["close"].pct_change().abs()
        n_jumps    = (pct_change > self.MAX_PRICE_CHANGE_PCT).sum()
        if n_jumps > 0:
            report.add_warning(
                f"{n_jumps} single-day price changes > {self.MAX_PRICE_CHANGE_PCT:.0%}."
                " Check for unadjusted splits or bad data."
            )

    def _check_volume_positive(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        if "volume" in df.columns:
            n_zero = (df["volume"] <= 0).sum()
            if n_zero > 0:
                report.add_warning(
                    f"Column 'volume' has {n_zero} zero/negative values."
                )

    def _check_sentiment_range(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        if "avg_sentiment" in df.columns:
            out_of_range = ((df["avg_sentiment"] < -1) | (df["avg_sentiment"] > 1)).sum()
            if out_of_range > 0:
                report.add_error(
                    f"'avg_sentiment' has {out_of_range} values outside [-1, 1]."
                )

    def _check_no_infinite(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        numeric = df.select_dtypes(include=[np.number])
        n_inf   = np.isinf(numeric.values).sum()
        if n_inf > 0:
            report.add_error(f"{n_inf} infinite values found in DataFrame.")

    def _check_constant_columns(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Constant columns add no signal — warn so they can be dropped."""
        numeric  = df.select_dtypes(include=[np.number])
        constant = [c for c in numeric.columns if numeric[c].nunique() <= 1]
        if constant:
            report.add_warning(
                f"Constant columns (zero variance): {constant}."
                " These will be dropped in feature pipeline."
            )

    def _add_stats(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Add summary stats to the report."""
        report.stats = {
            "rows":       len(df),
            "columns":    len(df.columns),
            "date_start": str(df.index.min().date()) if isinstance(df.index, pd.DatetimeIndex) else "n/a",
            "date_end":   str(df.index.max().date()) if isinstance(df.index, pd.DatetimeIndex) else "n/a",
            "nan_total":  int(df.isnull().sum().sum()),
        }

    def _finalise(self, report: ValidationReport) -> ValidationReport:
        """Log summary and optionally raise on errors."""
        if report.passed:
            logger.info(
                f"[validator] {report.name}: PASSED "
                f"({report.stats.get('rows', '?')} rows, "
                f"{len(report.warnings)} warnings)"
            )
        else:
            logger.error(
                f"[validator] {report.name}: FAILED "
                f"({len(report.errors)} errors)"
            )
            if self.strict:
                raise ValueError(
                    f"Validation failed for '{report.name}':\n"
                    + "\n".join(report.errors)
                )
        return report