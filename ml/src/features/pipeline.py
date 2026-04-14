"""
pipeline.py
-----------
Orchestrates the full feature engineering pipeline.
Calls TechnicalFeatures, MacroFeatures, SentimentFeatures
and merges everything into one clean DataFrame ready for causal discovery
and model training.

Also handles:
    - Aligning all data to the same trading day index
    - Forward-filling macro/sentiment on trading days with no updates
    - Dropping constant columns
    - Saving the final feature matrix to data/processed/features/

Usage:
    from ml.src.features.pipeline import FeaturePipeline
    pipeline = FeaturePipeline()

    # Training — reads from data/raw/, saves to data/processed/
    df = pipeline.build(ticker="AAPL")

    # Inference — builds a single-row feature vector from live data
    row = pipeline.build_live(ticker="AAPL", live_data=live_dict)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ml.src.data.loader import DataLoader, _load_config
from ml.src.data.validator import DataValidator
from ml.src.features.technical import TechnicalFeatures
from ml.src.features.macro import MacroFeatures
from ml.src.features.sentiment import SentimentFeatures
from ml.src.features.sector import SectorFeatures
from ml.src.features.earnings import EarningsFeatures
from ml.src.features.options import OptionsFeatures
from ml.src.features.finbert import FinBERTSentiment

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Full feature engineering pipeline.

    build(ticker)         → historical feature matrix (training)
    build_live(ticker)    → single-row feature vector (inference)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg       = _load_config(config_path)
        self.root      = Path(__file__).resolve().parents[3]
        self.loader    = DataLoader(config_path)
        self.validator = DataValidator(strict=False)   # warn, don't raise in pipeline
        self.tech      = TechnicalFeatures(config_path)
        self.macro     = MacroFeatures(config_path)
        self.sentiment = SentimentFeatures(config_path)
        self.sector    = SectorFeatures(config_path)
        self.earnings  = EarningsFeatures(config_path)
        self.options   = OptionsFeatures(config_path)
        self.finbert   = FinBERTSentiment(config_path)

        proc = self.cfg["data"]["processed_dir"]
        self.features_dir = self.root / proc / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        self.target_col  = self.cfg["model"]["target"]          # log_return_5d
        self.min_rows    = self.cfg["features"]["min_rows_after_engineering"]

    # -----------------------------------------------------------------------
    # Public — Training
    # -----------------------------------------------------------------------

    def build(self, ticker: str, force: bool = False) -> pd.DataFrame:
        """
        Build the full historical feature matrix for a ticker.

        Steps:
            1. Load raw prices, macro, sentiment from disk
            2. Validate each source
            3. Compute technical features
            4. Compute macro features
            5. Compute sentiment features
            6. Merge all on date index
            7. Forward-fill + drop NaN
            8. Validate final matrix
            9. Save to data/processed/features/{ticker}_features.csv

        Args:
            ticker: e.g. "AAPL"
            force:  if True, rebuild even if file already exists

        Returns:
            DataFrame with all features + target column
        """
        ticker   = ticker.upper()
        out_path = self.features_dir / f"{ticker}_features.csv"

        if out_path.exists() and not force:
            logger.info(f"[pipeline] {ticker}: feature matrix already exists, loading.")
            df = pd.read_csv(out_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df
            logger.info(f"[pipeline] {ticker}: cached file empty, rebuilding...")

        logger.info(f"[pipeline] Building feature matrix for {ticker} ...")

        # 1. Load raw data
        price_df     = self.loader.read_prices(ticker)
        macro_dict   = self._load_all_macro()
        sentiment_df = self._load_sentiment(ticker)

        # 2. Validate sources
        self.validator.validate_prices(price_df, ticker)

        # 3. Technical features — pass SPY for excess return target
        spy_close = None
        try:
            spy_df    = self.loader.read_macro("^GSPC")
            spy_close = spy_df["close"] if "close" in spy_df.columns else None
        except FileNotFoundError:
            logger.warning("[pipeline] SPY not found — using raw log return as target")
        self.tech.set_spy_close(spy_close)
        tech_df = self.tech.compute(price_df)

        # 4. Macro features
        macro_df = self.macro.compute(macro_dict)

        # 4b. Sector features
        spy_df    = None
        sect_df   = None
        try:
            spy_df  = self.loader.read_macro("^GSPC")
            sect_df = self.loader.read_macro("XLK")
        except FileNotFoundError:
            pass
        sector_df = self.sector.compute(price_df, spy_df, sect_df)

        # 4c. Earnings features
        earnings_df = self.earnings.compute(ticker, price_df)

        # 4d. Options IV features (historical proxy)
        options_df = self.options.compute_historical(ticker, price_df)

        # 5. Sentiment features
        if sentiment_df is not None and not sentiment_df.empty:
            self.validator.validate_sentiment(sentiment_df, ticker)
            sent_df = self.sentiment.compute(sentiment_df)
        else:
            sent_df = self.sentiment.fill_missing(
                pd.DataFrame(), index=tech_df.index
            )

        # 6. Merge on date index
        df = self._merge(tech_df, macro_df, sent_df, sector_df, earnings_df, options_df)

        # 7. Clean up
        df = self._clean(df)

        # 8. Validate final matrix
        report = self.validator.validate_feature_matrix(df, name=f"{ticker} features")
        if not report.passed:
            logger.error(f"[pipeline] Feature matrix validation failed for {ticker}.")

        if len(df) < self.min_rows:
            raise ValueError(
                f"[pipeline] Only {len(df)} rows after engineering — "
                f"need at least {self.min_rows}. Check data range."
            )

        # 9. Save
        df.to_csv(out_path)
        logger.info(
            f"[pipeline] Saved {len(df)} rows × {len(df.columns)} cols "
            f"→ {out_path.name}"
        )
        return df

    # -----------------------------------------------------------------------
    # Public — Inference
    # -----------------------------------------------------------------------

    def build_live(
        self, ticker: str, live_data: dict[str, pd.DataFrame]
    ) -> pd.Series:
        """
        Build a single-row feature vector for inference from live data.

        Args:
            ticker:    e.g. "AAPL"
            live_data: dict returned by DataLoader.load_live(ticker)
                       keys: "prices", "macro", "sentiment"

        Returns:
            pd.Series — one row of features, same columns as training matrix
                        target column (log_return_5d) is NOT included
        """
        ticker = ticker.upper()
        logger.info(f"[pipeline] Building live feature vector for {ticker} ...")

        price_df     = live_data.get("prices", pd.DataFrame())
        sentiment_df = live_data.get("sentiment", pd.DataFrame())

        # Build macro_dict from live macro DataFrame
        macro_wide = live_data.get("macro", pd.DataFrame())
        macro_dict = self._split_macro_wide(macro_wide)

        # Technical
        tech_df = self.tech.compute(price_df)

        # Macro
        macro_df = self.macro.compute(macro_dict) if macro_dict else pd.DataFrame()

        # Sentiment
        if sentiment_df is not None and not sentiment_df.empty:
            sent_df = self.sentiment.compute(sentiment_df)
        else:
            sent_df = self.sentiment.fill_missing(
                pd.DataFrame(), index=tech_df.index
            )

        # Sector features
        spy_df  = macro_dict.get("^GSPC", pd.DataFrame())
        sect_df = macro_dict.get("XLK",  pd.DataFrame())
        if not spy_df.empty:
            spy_df = spy_df.rename(columns={"close": "close"}) if "close" in spy_df.columns else spy_df
        sector_df = self.sector.compute(price_df, spy_df if not spy_df.empty else None,
                                        sect_df if not sect_df.empty else None)

        # Earnings features (live)
        earnings_df = self.earnings.compute(ticker, price_df)

        # Options IV features (live — uses current options chain)
        options_live = self.options.compute_live(ticker, price_df)
        # Corrected: do NOT broadcast live IV to historical rows
# Only use live IV for the LAST row (current date)
        from ml.src.features.options import OptionsFeatures
        OptionsFeatures.check_no_live_broadcast(options_live, context="pipeline.build_live")
# Then join only to the single-row live vector, not to the full history if not options_live.empty else None

        # Merge
        df = self._merge(tech_df, macro_df, sent_df, sector_df, earnings_df, options_df)
        df = self._clean(df, live=True)   # live=True: fill NaN, never drop columns

        # Drop target column — not available at inference time
        df = df.drop(columns=[self.target_col], errors="ignore")
        df = df.drop(columns=["log_return_5d"], errors="ignore")

        # Return last row — most recent date
        row = df.iloc[-1]
        logger.info(f"[pipeline] Live feature vector built: {len(row)} features.")
        return row

    # -----------------------------------------------------------------------
    # Public — Feature list
    # -----------------------------------------------------------------------

    def feature_columns(self, ticker: str) -> list[str]:
        """
        Return the list of feature columns for a ticker (excludes target).
        Reads from saved feature matrix — must call build() first.
        """
        path = self.features_dir / f"{ticker.upper()}_features.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No feature matrix for {ticker}. Run pipeline.build('{ticker}') first."
            )
        df = pd.read_csv(path, index_col=0, nrows=0)
        return [c for c in df.columns if c != self.target_col]

    # -----------------------------------------------------------------------
    # Private — Merge
    # -----------------------------------------------------------------------

    def _merge(
        self,
        tech_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        sent_df: pd.DataFrame,
        sector_df: pd.DataFrame = None,
        earnings_df: pd.DataFrame = None,
        options_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Merge technical, macro, sentiment, sector on the trading day index.
        Uses outer join then forward-fills macro and sentiment
        (they don't update every single day).
        """
        df = tech_df.copy()

        if macro_df is not None and not macro_df.empty:
            df = df.join(macro_df, how="left")

        if sent_df is not None and not sent_df.empty:
            df = df.join(sent_df, how="left")

        if sector_df is not None and not sector_df.empty:
            df = df.join(sector_df, how="left")

        if earnings_df is not None and not earnings_df.empty:
            df = df.join(earnings_df, how="left")

        if options_df is not None and not options_df.empty:
            df = df.join(options_df, how="left")

        # Forward-fill macro + sentiment (gaps on weekends/holidays)
        # Limit to 5 days to avoid propagating stale data across weekends + holidays
        macro_cols = macro_df.columns.tolist() if (macro_df is not None and not macro_df.empty) else []
        sent_cols  = sent_df.columns.tolist()  if (sent_df  is not None and not sent_df.empty)  else []
        ffill_cols = macro_cols + sent_cols

        if ffill_cols:
            df[ffill_cols] = df[ffill_cols].ffill(limit=5)

        return df

    # -----------------------------------------------------------------------
    # Private — Clean
    # -----------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame, live: bool = False) -> pd.DataFrame:
        """
        Post-merge cleaning.

        live=False (training): drop high-NaN + constant columns
        live=True  (inference): NEVER drop columns — fill NaN with 0 instead
                                so live feature vector always matches training columns
        """
        # Always remove forward-looking return columns except the configured target
        # This prevents leakage where a different return column remains as a feature
        # while the configured target is later dropped.
        return_cols = [
            c for c in df.columns
            if c.startswith("log_return_") or c.startswith("excess_return_")
        ]
        for c in return_cols:
            if c != self.target_col:
                df = df.drop(columns=[c], errors="ignore")

        if not live:
            # Training mode — drop last N rows (target not available)
            horizon = self.cfg["model"]["horizon_days"]
            df = df.iloc[:-horizon]

            # Drop high-NaN columns
            nan_ratio = df.isnull().mean()
            bad_cols  = nan_ratio[nan_ratio > 0.01].index.tolist()
            if bad_cols:
                logger.warning(f"[pipeline] Dropping high-NaN columns: {bad_cols}")
                df = df.drop(columns=bad_cols)

            # Drop constant columns
            numeric  = df.select_dtypes(include=[np.number])
            constant = [c for c in numeric.columns if numeric[c].nunique() <= 1]
            if constant:
                logger.warning(f"[pipeline] Dropping constant columns: {constant}")
                df = df.drop(columns=constant)

            # Drop duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            # Live inference mode — fill NaN instead of dropping
            # This ensures live vector always has same columns as training
            nan_cols = df.columns[df.isnull().any()].tolist()
            if nan_cols:
                logger.warning(
                    f"[pipeline] Live mode: filling {len(nan_cols)} NaN columns with 0 "
                    f"(insufficient history for: {nan_cols[:3]}{'...' if len(nan_cols) > 3 else ''})")
            df = df.loc[:, ~df.columns.duplicated()]

        # Fill remaining NaN with 0
        df = df.fillna(0)
        return df

    # -----------------------------------------------------------------------
    # Private — Helpers
    # -----------------------------------------------------------------------

    def _load_all_macro(self) -> dict[str, pd.DataFrame]:
        """Load all macro + sector ETF DataFrames from disk."""
        macro_tickers = (
            self.cfg["data"]["tickers"]["macro"]
            + self.cfg["data"]["tickers"]["sector_etfs"]
        )
        result = {}
        for symbol in macro_tickers:
            try:
                result[symbol] = self.loader.read_macro(symbol)
            except FileNotFoundError:
                logger.warning(f"[pipeline] Macro data for {symbol} not found, skipping.")
        return result

    def _load_sentiment(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load sentiment DataFrame from disk, return None if not found."""
        try:
            return self.loader.read_sentiment(ticker)
        except FileNotFoundError:
            logger.warning(f"[pipeline] No sentiment data for {ticker}.")
            return None

    def _split_macro_wide(
        self, macro_wide: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """
        Convert wide macro DataFrame (one column per symbol) back to
        the dict format expected by MacroFeatures.compute().
        Reverses the safe_filename mapping used by DataLoader.
        """
        if macro_wide is None or macro_wide.empty:
            return {}

        result = {}
        for col in macro_wide.columns:
            # Column name is already the original symbol (e.g. ^VIX, ^GSPC)
            df = macro_wide[[col]].rename(columns={col: "close"})
            result[col] = df
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build feature matrix for a ticker")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. AAPL")
    parser.add_argument("--force",  action="store_true", help="Rebuild even if exists")
    args = parser.parse_args()

    pipeline = FeaturePipeline()
    df = pipeline.build(args.ticker, force=args.force)
    print(f"\nFeature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range:     {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Saved to:       data/processed/features/{args.ticker.upper()}_features.csv")