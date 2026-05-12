"""
regime_characteriser.py
-----------------------
Produces Table 1 of the paper — showing WHY each Indian market regime is
structurally distinct. This provides the India-specific narrative support
that justifies treating each regime as a separate evaluation window.

Paper section: Section 3.1 — "Indian Market Regime Characterisation"

For each regime computes:
  - Price/return statistics (mean return, volatility, drawdown)
  - Macro statistics (VIX, P/E, P/B, sentiment)
  - Market structure metrics (days, VIX spikes, negative return days)

Usage:
    from ml.src.evaluation.regime_characteriser import RegimeCharacteriser
    char = RegimeCharacteriser(market='india')
    table = char.characterise(df, ticker='NIFTY')
    char.explain_regimes()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.evaluation.regime_splitter import RegimeSplitter

logger = logging.getLogger(__name__)


# India regime narrative context — used by explain_regimes()
INDIA_REGIME_NARRATIVES = {
    "gfc_recovery": (
        "GFC Recovery (2009-04 → 2013-12): Nifty recovered from 3,060 to 6,364 (+106%) "
        "driven by global quantitative easing and FII inflows. CAVEAT: only 266 training "
        "rows precede this regime, making LightGBM unreliable (needs ≥500). DA ~0.46 is "
        "expected. This is a data availability limitation, not a model failure."
    ),
    "pre_demonetisation": (
        "Pre-Demonetisation Bull (2014-01 → 2016-10): Modi election rally drove Nifty +39%. "
        "P/E expanded from 19 to 23 on reform expectations. VIX averaged 15.5, indicating "
        "low fear. Data gaps in 2014-15 reduce available rows to ~371. The 1,407 training "
        "rows before this regime are sufficient for reliable model estimation."
    ),
    "demonetisation": (
        "Demonetisation Shock (2016-11 → 2017-02): Modi banned ₹500/₹1000 notes on "
        "8 Nov 2016. BOTH price data AND headlines are absent from the source files for "
        "this period — this is a data collection gap, not a loader bug. The regime is "
        "documented but EXCLUDED from backtesting (0 test rows)."
    ),
    "recovery_india": (
        "Post-Demonetisation Recovery (2017-03 → 2019-12): Market absorbed the cash "
        "crunch shock. GST rollout (July 2017) introduced new macro uncertainty. Nifty "
        "consolidated between 9,000-12,000. This is a normalisation period where "
        "fundamental signals (P/E mean-reversion) should dominate over crisis indicators."
    ),
    "covid_crash": (
        "COVID Crash (2020-01 → 2020-06): India VIX peaked at 79.3 (mean ~38), the "
        "highest in the dataset. Nifty fell 38% in 6 weeks during March 2020. This is the "
        "hardest regime to predict — any model claiming DA > 0.60 here likely has "
        "price-level leakage. Causal features should activate VIX-based signals strongly."
    ),
    "recovery_covid": (
        "Stimulus Melt-Up (2020-07 → 2021-12): Nifty surged +64% on RBI rate cuts and "
        "global liquidity. P/E peaked at 42 — the highest valuation in the dataset. "
        "This regime is hard because the market kept rising against fundamental signals "
        "(high PE, rising VIX). Momentum features should outperform value signals here."
    ),
    "rate_hike_rbi": (
        "RBI Rate Hike Cycle (2022-01 → 2023-06): RBI raised rates by 250bps to fight "
        "inflation. VIX was elevated at 23.4 (vs 15 in calm periods). Model performs well "
        "here — rate hike regimes have clear causal structure: yield signals and VIX "
        "changes have strong predictive power for equity returns."
    ),
    "current_bull": (
        "Current India Bull Run (2023-07 → 2024-11): Growth narrative driven by "
        "manufacturing capex and AI optimism. Low VIX (15.4), moderate P/E (22.6). "
        "Model's best regime — most similar to the overall training data distribution, "
        "so global causal features generalise well here."
    ),
}


class RegimeCharacteriser:
    """
    Computes structural statistics for each market regime and provides
    narrative context for the paper.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        market: str = "india",
    ):
        self.cfg = _load_config(config_path)
        self.market = market
        self.splitter = RegimeSplitter(config_path, market=market)

    # -----------------------------------------------------------------------
    # Public — characterise regimes
    # -----------------------------------------------------------------------

    def characterise(
        self,
        df: pd.DataFrame,
        ticker: str,
        market: str = "india",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Compute structural statistics for each regime.

        Args:
            df:     Feature matrix with DatetimeIndex.
            ticker: Stock ticker.
            market: Market identifier.
            save:   If True, save to paper_output/{ticker}/.

        Returns:
            DataFrame with regimes as rows, statistics as columns.
        """
        logger.info(f"[regime_char] Computing regime characterisation for {ticker} ...")

        regime_splits = self.splitter.split_all(df)
        rows = []

        for regime_name, regime_df in regime_splits.items():
            row = self._compute_regime_stats(regime_name, regime_df)
            rows.append(row)

        if not rows:
            logger.warning("[regime_char] No regimes with data found.")
            return pd.DataFrame()

        result = pd.DataFrame(rows).set_index("regime")

        # Print formatted table
        self._print_table(result, ticker)

        # Save
        if save:
            root = Path(__file__).resolve().parents[3]
            output_dir = root / "paper_output" / ticker.upper()
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"table1_regime_characterisation.csv"
            result.to_csv(out_path)
            logger.info(f"[regime_char] Saved → {out_path}")

        return result

    # -----------------------------------------------------------------------
    # Public — narrative explanation
    # -----------------------------------------------------------------------

    def explain_regimes(self) -> None:
        """
        Print a narrative paragraph per regime explaining what the statistics
        mean in Indian market context.
        """
        W = 80
        print(f"\n{'='*W}")
        print(f"  REGIME NARRATIVES — INDIA MARKET CONTEXT")
        print(f"{'='*W}")

        for regime_name in self.splitter.regime_names:
            narrative = INDIA_REGIME_NARRATIVES.get(regime_name)
            if narrative:
                # Word-wrap at 76 chars for clean printing
                wrapped = self._wrap_text(narrative, width=76)
                print(f"\n  {wrapped}")
            else:
                dates = self.splitter.regime_dates.get(regime_name, ("?", "?"))
                print(f"\n  {regime_name} ({dates[0]} → {dates[1]}): No narrative available.")

        print(f"\n{'='*W}\n")

    # -----------------------------------------------------------------------
    # Private — compute stats for one regime
    # -----------------------------------------------------------------------

    def _compute_regime_stats(
        self, regime_name: str, regime_df: pd.DataFrame
    ) -> dict:
        """Compute all statistics for a single regime."""
        row = {
            "regime": regime_name,
            "n_rows": len(regime_df),
            "date_start": str(regime_df.index.min().date()) if len(regime_df) > 0 else "N/A",
            "date_end": str(regime_df.index.max().date()) if len(regime_df) > 0 else "N/A",
        }

        # ── Price/Return statistics ─────────────────────────────────────
        row.update(self._safe_stat(regime_df, "log_return_5d", "mean", "mean_return"))
        row.update(self._safe_stat(regime_df, "log_return_5d", "std", "std_return"))

        # Annualised volatility
        std_ret = row.get("std_return")
        if std_ret is not None and not np.isnan(std_ret):
            row["annualised_vol"] = round(std_ret * np.sqrt(252), 4)
        else:
            row["annualised_vol"] = np.nan

        # Total return (sum of daily log returns)
        row.update(self._safe_stat(regime_df, "log_return_1d", "sum", "total_return"))

        # Max drawdown
        row["max_drawdown"] = self._compute_max_drawdown(regime_df)

        # ── Macro statistics ────────────────────────────────────────────
        row.update(self._safe_stat(regime_df, "india_vix", "mean", "mean_vix"))
        row.update(self._safe_stat(regime_df, "pe_ratio", "mean", "mean_pe"))
        row.update(self._safe_stat(regime_df, "pb_ratio", "mean", "mean_pb"))

        # Sentiment — try multiple column names
        for sent_col in ["sentiment_score", "precomputed_sentiment"]:
            if sent_col in regime_df.columns:
                row.update(self._safe_stat(regime_df, sent_col, "mean", "mean_sentiment"))
                break
        else:
            row["mean_sentiment"] = np.nan

        # VIX regime percentage
        if "vix_regime" in regime_df.columns:
            try:
                row["vix_regime_pct"] = round(float((regime_df["vix_regime"] == 1).mean()), 4)
            except Exception:
                row["vix_regime_pct"] = np.nan
        else:
            row["vix_regime_pct"] = np.nan

        # ── Market structure ────────────────────────────────────────────
        # VIX spike days (India VIX > 30 = extreme fear)
        if "india_vix" in regime_df.columns:
            try:
                row["vix_spike_days"] = int((regime_df["india_vix"] > 30).sum())
            except Exception:
                row["vix_spike_days"] = np.nan
        else:
            row["vix_spike_days"] = np.nan

        # Negative return days
        if "log_return_1d" in regime_df.columns:
            try:
                row["negative_return_days"] = round(
                    float((regime_df["log_return_1d"] < 0).mean()), 4
                )
            except Exception:
                row["negative_return_days"] = np.nan
        else:
            row["negative_return_days"] = np.nan

        return row

    def _safe_stat(
        self,
        df: pd.DataFrame,
        col: str,
        agg: str,
        output_name: str,
    ) -> dict:
        """Safely compute a statistic, returning NaN if column missing."""
        if col not in df.columns:
            return {output_name: np.nan}
        try:
            val = getattr(df[col].dropna(), agg)()
            return {output_name: round(float(val), 4)}
        except Exception:
            return {output_name: np.nan}

    def _compute_max_drawdown(self, df: pd.DataFrame) -> float:
        """Compute maximum drawdown from daily log returns."""
        if "log_return_1d" not in df.columns:
            return np.nan
        try:
            cum_returns = (1 + df["log_return_1d"]).cumprod()
            peak = cum_returns.cummax()
            drawdown = (cum_returns - peak) / peak
            return round(float(drawdown.min()), 4)
        except Exception:
            return np.nan

    # -----------------------------------------------------------------------
    # Private — display
    # -----------------------------------------------------------------------

    def _print_table(self, result: pd.DataFrame, ticker: str) -> None:
        """Print the characterisation table in a readable format."""
        W = 100
        print(f"\n{'='*W}")
        print(f"  TABLE 1 — REGIME CHARACTERISATION: {ticker.upper()}")
        print(f"{'='*W}")

        # Select key columns for display
        display_cols = [
            "n_rows", "date_start", "date_end",
            "mean_return", "annualised_vol", "max_drawdown", "total_return",
            "mean_vix", "mean_pe", "negative_return_days", "vix_spike_days",
        ]
        available = [c for c in display_cols if c in result.columns]

        # Print in two blocks for readability
        # Block 1: Returns
        return_cols = [c for c in ["n_rows", "date_start", "date_end",
                                    "mean_return", "annualised_vol",
                                    "max_drawdown", "total_return"] if c in available]
        if return_cols:
            print(f"\n  Price & Return Statistics:")
            print(result[return_cols].to_string())

        # Block 2: Macro
        macro_cols = [c for c in ["mean_vix", "mean_pe", "mean_pb",
                                   "mean_sentiment", "vix_regime_pct",
                                   "vix_spike_days", "negative_return_days"] if c in available]
        if macro_cols:
            print(f"\n  Macro & Structure Statistics:")
            print(result[macro_cols].to_string())

        print(f"\n{'='*W}")

    @staticmethod
    def _wrap_text(text: str, width: int = 76) -> str:
        """Simple word wrapping for narrative paragraphs."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > width:
                lines.append(current_line)
                current_line = "  " + word  # indent continuation
            else:
                current_line = current_line + " " + word if current_line else word
        if current_line:
            lines.append(current_line)
        return "\n  ".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Regime characterisation table (Paper Table 1)"
    )
    parser.add_argument("--ticker", type=str, required=True, help="e.g. NIFTY")
    parser.add_argument("--market", type=str, default="india", choices=["us", "india"])
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg = _load_config()

    # Load feature matrix
    if ticker in ("NIFTY", "^NSEI", "NIFTY50") or args.market == "india":
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    char = RegimeCharacteriser(market=args.market)
    table = char.characterise(df, ticker, market=args.market)
    char.explain_regimes()
