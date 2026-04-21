"""
nifty_loader.py
---------------
Loads and preprocesses all Indian market data files:

    Price + Fundamental data:
        Unused_Data.csv       — 2008-01-28 to 2018-12-27
                                Open + Close + P/E + P/B + VIX + Sentiment_Score
                                No Volume, no High/Low columns
        Final_Data.csv        — 2019-01-01 to 2024-11-01
                                Open + Close + Volume + P/E + P/B + VIX + Sentiment_Score

    Headlines (sentiment):
        financial_headlines_with_serial_no_2.csv — 2008-2018, date format DD-MM-YYYY
        financial_headlines_with_serial_no.csv   — 2019-2024, date format D-MM-YY

    KNOWN DATA GAPS IN UNUSED_DATA (all handled gracefully):
        2014-10-31 to 2015-11-02  (entire 2015 Jan-Oct missing, ~250 trading days)
        2016-10-28 to 2017-03-01  (demonetisation shock period, Nov 2016-Feb 2017)
        Several ~30-day monthly gaps in 2014, 2016, 2018

    DEMONETISATION REGIME NOTE:
        Both price data AND headlines for Nov 2016-Feb 2017 are absent from
        the source files. This is a data collection gap, not a loader bug.
        The demonetisation regime in config.yaml is kept for paper documentation
        but is EXCLUDED from Option B backtesting (min_test_samples enforced).
        The regime_splitter will find 0 rows for that period and skip it cleanly.

    SENTIMENT NORMALIZATION:
        Unused_Data Sentiment_Score: raw range [-2.9, +12.7], mean 1.45, std 1.97
        Final_Data  Sentiment_Score: raw range [-2.5, +19.9], mean 4.96, std 4.14
        These are DIFFERENT scales from different data collection periods.
        Each file is independently z-score normalized to [-1, +1] before joining.
        This prevents the Final_Data's higher magnitudes from dominating the model.

Combined coverage: 2008-2024 (~3,555 rows, 69,000+ headlines)

Store files at: ml/data/raw/sentiment/

Usage:
    from ml.src.data.nifty_loader import NiftyLoader
    loader = NiftyLoader()
    df     = loader.build_feature_matrix()
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.features.nifty_feature_guard import apply_nifty_feature_guard_to_pipeline

logger = logging.getLogger(__name__)


class NiftyLoader:
    """
    Loads all Indian market data files and builds a feature matrix
    compatible with the existing PCMCI + ensemble pipeline.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg      = _load_config(config_path)
        self.root     = Path(__file__).resolve().parents[3]
        self.data_dir = self.root / self.cfg["data"]["raw_dir"] / "sentiment"

        self.final_data_path  = self.data_dir / "Final_Data.csv"
        self.unused_data_path = self.data_dir / "Unused_Data.csv"

        # 2008-2018 headlines: DD-MM-YYYY format  (e.g. 21-01-2008)
        self.headlines_old = self.data_dir / "financial_headlines_with_serial_no_2.csv"
        # 2019-2024 headlines: D-MM-YY format     (e.g. 1-01-19)
        self.headlines_new = self.data_dir / "financial_headlines_with_serial_no.csv"

        self.out_dir = self.root / self.cfg["data"]["processed_dir"] / "features"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        """
        Load Nifty 50 price data from both CSVs combined (2008-2024).

        Unused_Data (2008-2018):
          - Has Open and Close only (no High, Low, or Volume in source)
          - High  = max(Open, Close) * 1.005  (conservative intraday range proxy)
          - Low   = min(Open, Close) * 0.995
          - Volume = 0.0  (not available; volume-based features gracefully degrade
                           but are NOT removed — they carry non-zero signal for
                           2019-2024 and zero-signal for 2008-2018, which is
                           informative for the regime-aware model)

        Final_Data (2019-2024):
          - Has Open, Close, and Volume
          - High/Low approximated identically (source lacks true intraday H/L)

        Returns:
            DataFrame with DatetimeIndex, columns: open, high, low, close, volume
        """
        # PAPER DISCLOSURE REQUIRED: True intraday High/Low not available for 2008-2018.
        # H/L are approximated as open/close ± 0.5%. This compresses ATR and bb_width
        # for the pre-2019 period. All technical indicators dependent on H/L are
        # unreliable for 2008-2018. See paper Section 3.1 for robustness analysis.
        import warnings
        warnings.warn(
            "Nifty 2008-2018: High/Low approximated from Open/Close ±0.5%. "
            "ATR, Bollinger width, and vol features are unreliable for this period. "
            "Paper must disclose this as a threat to validity.",
            UserWarning, stacklevel=2
        )
        frames = []

        # ── Unused_Data: 2008-01-28 → 2018-12-27 ─────────────────────────
        if self.unused_data_path.exists():
            df_old = pd.read_csv(self.unused_data_path, parse_dates=["Date"])
            df_old = df_old.set_index("Date").sort_index()
            df_old.index = pd.to_datetime(df_old.index)

            old = pd.DataFrame(index=df_old.index)
            old["open"]   = pd.to_numeric(df_old["Open"],  errors="coerce")
            old["close"]  = pd.to_numeric(df_old["Close"], errors="coerce")
            old["volume"] = 0.0
            old["high"]   = old[["open", "close"]].max(axis=1) * 1.005
            old["low"]    = old[["open", "close"]].min(axis=1) * 0.995
            old = old.dropna(subset=["open", "close"])
            old = old[old["close"] > 0]
            frames.append(old)
            logger.info(
                f"[nifty_loader] Unused_Data: {len(old)} rows "
                f"({old.index.min().date()} → {old.index.max().date()})"
            )
        else:
            logger.warning(
                f"[nifty_loader] Unused_Data.csv not found at {self.unused_data_path}. "
                f"Coverage will start from 2019 only."
            )

        # ── Final_Data: 2019-01-01 → 2024-11-01 ──────────────────────────
        self._check_file(self.final_data_path)
        df_new = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df_new = df_new.set_index("Date").sort_index()
        df_new.index = pd.to_datetime(df_new.index)

        new = pd.DataFrame(index=df_new.index)
        new["open"]   = pd.to_numeric(df_new["Open"],   errors="coerce")
        new["close"]  = pd.to_numeric(df_new["Close"],  errors="coerce")
        new["volume"] = pd.to_numeric(df_new["Volume"], errors="coerce").fillna(0.0)
        new["high"]   = new[["open", "close"]].max(axis=1) * 1.005
        new["low"]    = new[["open", "close"]].min(axis=1) * 0.995
        new = new.dropna(subset=["open", "close"])
        new = new[new["close"] > 0]
        frames.append(new)
        logger.info(
            f"[nifty_loader] Final_Data: {len(new)} rows "
            f"({new.index.min().date()} → {new.index.max().date()})"
        )

        # ── Combine ────────────────────────────────────────────────────────
        price_df = pd.concat(frames).sort_index()
        price_df = price_df[~price_df.index.duplicated(keep="last")]

        logger.info(
            f"[nifty_loader] Combined prices: {len(price_df)} rows "
            f"({price_df.index.min().date()} → {price_df.index.max().date()})"
        )
        return price_df

    def load_fundamental_features(self) -> pd.DataFrame:
        """
        Load P/E, P/B, India VIX from both CSVs combined (2008-2024).
        Also reads pre-computed Sentiment_Score as a supplementary signal,
        normalized independently per file to account for scale differences.
        """
        frames = []

        if self.unused_data_path.exists():
            df_old = pd.read_csv(self.unused_data_path, parse_dates=["Date"])
            df_old = df_old.set_index("Date").sort_index()
            df_old.index = pd.to_datetime(df_old.index)
            # Normalize sentiment within this file's distribution only
            s = pd.to_numeric(df_old["Sentiment_Score"], errors="coerce").fillna(0.0)
            roll = s.rolling(window=252, min_periods=1)
            df_old["_sent_norm"] = ((s - roll.mean()) / (roll.std() + 1e-8)).clip(-3, 3) / 3
            frames.append(df_old)

        self._check_file(self.final_data_path)
        df_new = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df_new = df_new.set_index("Date").sort_index()
        df_new.index = pd.to_datetime(df_new.index)
        s = pd.to_numeric(df_new["Sentiment_Score"], errors="coerce").fillna(0.0)
        roll = s.rolling(window=252, min_periods=1)
        df_new["_sent_norm"] = ((s - roll.mean()) / (roll.std() + 1e-8)).clip(-3, 3) / 3
        frames.append(df_new)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        result = pd.DataFrame(index=df.index)

        # Raw fundamental values
        result["pe_ratio"]             = pd.to_numeric(df["P/E"], errors="coerce")
        result["pb_ratio"]             = pd.to_numeric(df["P/B"], errors="coerce")
        result["india_vix"]            = pd.to_numeric(df["Vix"], errors="coerce")
        result["precomputed_sentiment"] = df["_sent_norm"]

        # 1-day and 5-day changes
        result["pe_change_1d"]  = result["pe_ratio"].diff(1)
        result["pb_change_1d"]  = result["pb_ratio"].diff(1)
        result["vix_change_1d"] = result["india_vix"].diff(1)
        result["vix_change_5d"] = result["india_vix"].diff(5)

        # PE momentum / acceleration — helps disambiguate high PE driven by
        # valuation vs momentum (PE rising quickly often signals a different
        # regime than persistently high PE)
        result["pe_change_5d"]   = result["pe_ratio"].diff(5)
        result["pe_vs_1y_high"]  = (
            result["pe_ratio"]
            / result["pe_ratio"].rolling(252, min_periods=60).max()
        )
        result["pe_accel"]       = result["pe_change_5d"].diff(5)

        # Rolling moving averages (min_periods avoids NaN at the start of the series)
        result["vix_ma_5"]  = result["india_vix"].rolling(5,  min_periods=1).mean()
        result["vix_ma_10"] = result["india_vix"].rolling(10, min_periods=1).mean()
        result["pe_ma_10"]  = result["pe_ratio"].rolling(10,  min_periods=1).mean()
        result["pb_ma_10"]  = result["pb_ratio"].rolling(10,  min_periods=1).mean()

        # VIX-derived extreme / crash signals
        result["vix_delta_3d"]     = result["india_vix"].diff(3)
        result["vix_extreme"]      = (result["india_vix"] > 40).astype(float)
        result["vix_crash_signal"] = (
            (result["india_vix"] > 40) & (result["vix_delta_3d"] > 5)
        ).astype(float)

        # QE / liquidity signal (when VIX is falling from high levels but PE
        # remains elevated we may be in a liquidity-driven melt-up)
        result["vix_falling"] = (result["vix_change_5d"] < -2).astype(float)
        result["qe_signal"]    = (
            result["vix_falling"].astype(float) * (result["pe_ratio"] > 22).astype(float)
        ).astype(float)

        # Percentile ranks: where is today vs the past 252 trading days
        # min_periods=60 allows these to be computed after ~3 months of data
        result["vix_percentile"] = result["india_vix"].rolling(252, min_periods=60).rank(pct=True)
        result["pe_percentile"]  = result["pe_ratio"].rolling(252,  min_periods=60).rank(pct=True)
        result["pb_percentile"]  = result["pb_ratio"].rolling(252,  min_periods=60).rank(pct=True)

        # Regime signals: is current value above its 60-day average?
        result["pe_regime"]  = (
            result["pe_ratio"] > result["pe_ratio"].rolling(60, min_periods=20).mean()
        ).astype(float)
        result["vix_regime"] = (
            result["india_vix"] > result["india_vix"].rolling(60, min_periods=20).mean()
        ).astype(float)

        # Combined danger signal: high PE + high VIX (overvalued + fearful)
        result["pe_vix_danger"]    = result["pe_percentile"] * result["vix_percentile"]

        # Crisis detector: VIX > 30 is extreme fear on India scale
        result["vix_crisis"]       = (result["india_vix"] > 30).astype(float)

        # Valuation extremes
        result["market_expensive"] = (result["pe_ratio"] > 25).astype(float)
        result["market_cheap"]     = (result["pe_ratio"] < 15).astype(float)

        # Additional long-window valuation feature: 5-year PE mean deviation
        # computed here so it's available to downstream regime logic.
        result["pe_vs_5y_mean"] = (
            result["pe_ratio"]
            / result["pe_ratio"].rolling(1260, min_periods=252).mean()
            - 1.0
        )

        # VIX MA slope — measures whether stress is building or abating
        result["vix_ma20_slope"] = (
            result["india_vix"].rolling(20, min_periods=5).mean().diff(5)
        )

        logger.info(
            f"[nifty_loader] Fundamental features: {len(result.columns)} cols, "
            f"{len(result)} rows "
            f"({result.index.min().date()} → {result.index.max().date()})"
        )
        return result.fillna(0.0)

    def load_sentiment(self, use_finbert: bool = False) -> pd.DataFrame:
        """
        Load and score all Indian financial headlines (2008-2024).

        Date format differences between files:
            financial_headlines_with_serial_no_2.csv: DD-MM-YYYY (e.g. 21-01-2008)
            financial_headlines_with_serial_no.csv:   D-MM-YY    (e.g. 1-01-19)
        Both are correctly parsed by pandas with dayfirst=True.

        Gap handling:
            Headlines for Nov 2016-Feb 2017 (demonetisation) are absent from
            the source file — this mirrors the price data gap. Sentiment features
            for that period are zero-filled after ffill limit is exhausted.

        Returns:
            DataFrame with DatetimeIndex and sentiment feature columns, or
            falls back to pre-computed Sentiment_Score if no headline files found.
        """
        headlines_df = self._load_all_headlines()

        if headlines_df.empty:
            logger.warning(
                "[nifty_loader] No headlines loaded — "
                "falling back to pre-computed Sentiment_Score."
            )
            return self._load_precomputed_sentiment()

        # Score headlines
        from ml.src.features.finbert import FinBERTSentiment
        scorer = FinBERTSentiment()

        if use_finbert and scorer.is_available():
            logger.info("[nifty_loader] Scoring with FinBERT ...")
            scores = scorer.score_batch(headlines_df["headline"].tolist())
        else:
            logger.info("[nifty_loader] Scoring with keyword model ...")
            scores = [scorer._keyword_score(h) for h in headlines_df["headline"]]

        headlines_df = headlines_df.copy()
        headlines_df["score"] = scores

        # Aggregate to daily
        daily = (
            headlines_df
            .groupby("date")["score"]
            .agg(["mean", "std", "count"])
        )
        daily.columns = ["sentiment_score", "sentiment_std", "article_count"]
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()

        # Rolling features
        daily["sentiment_ma_3d"]    = daily["sentiment_score"].rolling(3,  min_periods=1).mean()
        daily["sentiment_ma_5d"]    = daily["sentiment_score"].rolling(5,  min_periods=1).mean()
        daily["sentiment_ma_10d"]   = daily["sentiment_score"].rolling(10, min_periods=1).mean()
        daily["sentiment_momentum"] = daily["sentiment_ma_3d"] - daily["sentiment_ma_10d"]
        daily["buzz_ma_3d"]         = daily["article_count"].rolling(3, min_periods=1).mean()
        daily["buzz_ma_5d"]         = daily["article_count"].rolling(5, min_periods=1).mean()
        daily["sentiment_std_5d"]   = (
            daily["sentiment_score"].rolling(5, min_periods=1).std().fillna(0.0)
        )

        logger.info(
            f"[nifty_loader] Sentiment loaded: {len(daily)} days, "
            f"{int(daily['article_count'].sum()):,} total headlines "
            f"({daily.index.min().date()} → {daily.index.max().date()})"
        )
        return daily.fillna(0.0)

    def build_feature_matrix(
        self,
        use_finbert: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build complete Nifty 50 feature matrix (2008-2024).
        Combines technical + fundamental + sentiment features.
        Ready for PCMCI causal discovery + model training.
        """
        from ml.src.features.technical import TechnicalFeatures

        logger.info("[nifty_loader] Building Nifty 50 feature matrix (2008-2024) ...")

        # 1. Load prices
        price_df = self.load_prices()

        # 2. Technical features — SAFE: compute_features() returns only
        #    backward-looking indicators (no shift(-5) targets).
        tech = TechnicalFeatures()
        tech_feat_df = tech.compute_features(price_df)
        # Attach ONLY the target column (log_return_5d) separately.
        # No SPY benchmark for index prediction, so spy_close=None.
        target_df = tech.compute_targets(price_df, spy_close=None)
        tech_df = tech_feat_df.join(target_df[["log_return_5d"]], how="left")

        # 3. Fundamental features (P/E, P/B, India VIX)
        fund_df = self.load_fundamental_features()
        fund_df = fund_df.reindex(tech_df.index).ffill(limit=5).fillna(0.0)

        # 3b. Close vs 200-day MA — computed here because we have the price
        # series available in this scope. Uses min_periods=60 (~3 months).
        try:
            close_vs_200 = (
                price_df["close"]
                / price_df["close"].rolling(200, min_periods=60).mean()
                - 1.0
            )
            close_vs_200 = close_vs_200.reindex(tech_df.index).ffill(limit=5).fillna(0.0)
            fund_df["close_vs_200ma"] = close_vs_200
        except Exception:
            # If anything goes wrong here, continue — feature is optional
            logger.warning("[nifty_loader] Could not compute close_vs_200ma.")

        # 4. Sentiment (headline-based or precomputed fallback)
        sent_df = self.load_sentiment(use_finbert=use_finbert)
        sent_df.index = pd.to_datetime(sent_df.index)
        # ffill limit=5 bridges weekends and short gaps (including the ~4-day
        # mini gaps in the data). For extremely long headline gaps (e.g.
        # demonetisation Nov 2016–Feb 2017) we DROP those dates from the
        # feature matrix rather than zero-fill misleadingly. This avoids
        # introducing a spurious "zero sentiment" signal during a major
        # market shock. We consider gaps > 30 consecutive days as long.
        sent_df = sent_df.reindex(tech_df.index).ffill(limit=5)

        # Identify long missing stretches (article_count == 0) and drop them
        missing = sent_df["article_count"].fillna(0.0) == 0
        to_drop = []
        mask = missing.values
        idxs = sent_df.index
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                run_len = j - i
                if run_len > 30:
                    to_drop.extend(idxs[i:j].tolist())
                i = j
            else:
                i += 1

        if to_drop:
            logger.info(
                f"[nifty_loader] Dropping {len(to_drop)} rows due to long headline gaps (>30d)"
            )
            # Drop these dates from the technical and fundamental frames so the
            # final feature matrix does not include the demonetisation gap.
            tech_df = tech_df.drop(index=to_drop, errors="ignore")
            fund_df = fund_df.drop(index=to_drop, errors="ignore")
            sent_df = sent_df.drop(index=to_drop, errors="ignore")

        # Fill remaining small gaps with conservative zeros after removal
        sent_df = sent_df.fillna(0.0)

        # 5. Merge
        df = tech_df.copy()
        df = df.join(fund_df, how="left")
        df = df.join(sent_df, how="left")
        # After: df = df.join(sent_df, how="left")
# Add:
        from ml.src.features.nifty_feature_guard import apply_nifty_feature_guard_to_pipeline
        df = apply_nifty_feature_guard_to_pipeline(df)
        
        # 6. Final cleanup
        # CRITICAL: Drop trailing rows where the forward-looking target is legitimately unavailable 
        # before any forward-filling to prevent corrupting the target column with stale returns.
        if "log_return_5d" in df.columns:
            df = df.dropna(subset=["log_return_5d"])
            
        # CRITICAL: Exclude target columns from ffill so future return values
        # don't get propagated across interior gaps.
        leaky_cols = [c for c in df.columns 
                      if (c.startswith("log_return_") and c != "log_return_1d") 
                      or c.startswith("excess_return_")]
        safe_cols = [c for c in df.columns if c not in leaky_cols]
        df[safe_cols] = df[safe_cols].ffill(limit=5)
        # AUDIT FIX: Do NOT fill target/leaky columns with zero — that creates
        # fabricated zero-return observations. Drop any remaining rows where the
        # target is still NaN (interior gaps), then zero-fill only safe columns.
        if "log_return_5d" in df.columns:
            remaining_nans = df["log_return_5d"].isna().sum()
            if remaining_nans > 0:
                logger.info(
                    f"[nifty_loader] Dropping {remaining_nans} rows with NaN target "
                    f"(interior gaps — not zero-filled to avoid fabricated returns)"
                )
                df = df.dropna(subset=["log_return_5d"])
        df[safe_cols] = df[safe_cols].fillna(0.0)
        # Leaky cols: drop remaining NaN rows rather than zero-fill
        for lc in leaky_cols:
            if lc in df.columns:
                df = df.dropna(subset=[lc])

        # Drop any perfectly constant columns (no variance = no signal)
        numeric = df.select_dtypes(include=["number"])
        constant_cols = [c for c in numeric.columns if numeric[c].nunique() <= 1]
        if constant_cols:
            logger.warning(
                f"[nifty_loader] Dropping {len(constant_cols)} constant columns: "
                f"{constant_cols}"
            )
            df = df.drop(columns=constant_cols)

        logger.info(
            f"[nifty_loader] Feature matrix complete: "
            f"{df.shape[0]} rows × {df.shape[1]} cols "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )

        # 7. Save
        if save:
            out_path = self.out_dir / "NIFTY_features.csv"
            df.to_csv(out_path)
            logger.info(f"[nifty_loader] Saved → {out_path.name}")

        return df

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_all_headlines(self) -> pd.DataFrame:
        """
        Load and combine both headline files into a single DataFrame.

        Date parsing strategy:
            Both files use day-first ordering but different formats:
              - 2008-2018 file: DD-MM-YYYY  (e.g. 21-01-2008)
              - 2019-2024 file: D-MM-YY     (e.g. 1-01-19)
            pandas with dayfirst=True handles both correctly via dateutil
            fallback. The UserWarning about format inference is suppressed
            because the output is verified correct by inspection and tests.
        """
        frames = []

        for path in [self.headlines_old, self.headlines_new]:
            if not path.exists():
                logger.warning(f"[nifty_loader] Not found: {path.name}")
                continue
            try:
                raw = pd.read_csv(path)[["date", "headline"]].dropna()

                # Suppress the "could not infer format" UserWarning
                # Both formats parse correctly with dayfirst=True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    raw["date"] = pd.to_datetime(
                        raw["date"],
                        dayfirst=True,
                        errors="coerce",
                    )

                before = len(raw)
                raw = raw.dropna(subset=["date"])
                raw["date"] = raw["date"].dt.normalize()
                dropped = before - len(raw)
                if dropped > 0:
                    logger.warning(
                        f"[nifty_loader] {path.name}: dropped {dropped} rows "
                        f"with unparseable dates."
                    )

                frames.append(raw)
                logger.info(
                    f"[nifty_loader] Loaded {len(raw):,} headlines from {path.name} "
                    f"({raw['date'].min().date()} → {raw['date'].max().date()})"
                )
            except Exception as e:
                logger.warning(f"[nifty_loader] Failed to load {path.name}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["date", "headline"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(
            f"[nifty_loader] Combined headlines: {len(combined):,} rows "
            f"({combined['date'].min().date()} → {combined['date'].max().date()})"
        )
        return combined

    def _load_precomputed_sentiment(self) -> pd.DataFrame:
        """
        Fallback: normalize and return the pre-computed Sentiment_Score
        columns from both CSV files.

        CRITICAL: independently normalizes each file before joining.
        Unused_Data: mean 1.45, std 1.97 → z-scored then /3 → [-1, +1]
        Final_Data:  mean 4.96, std 4.14 → z-scored then /3 → [-1, +1]
        Without this step, the 2019-2024 period's higher raw scores would
        look systematically more positive than 2008-2018, creating a
        spurious level-shift that the model would incorrectly learn as signal.
        """
        frames = []

        if self.unused_data_path.exists():
            df_old = pd.read_csv(self.unused_data_path, parse_dates=["Date"])
            df_old = df_old.set_index("Date").sort_index()
            df_old.index = pd.to_datetime(df_old.index)
            s = pd.to_numeric(df_old["Sentiment_Score"], errors="coerce").fillna(0.0)
            roll = s.rolling(window=252, min_periods=1)
            norm = ((s - roll.mean()) / (roll.std() + 1e-8)).clip(-3, 3) / 3
            frames.append(norm.rename("_s"))

        df_new = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df_new = df_new.set_index("Date").sort_index()
        df_new.index = pd.to_datetime(df_new.index)
        s = pd.to_numeric(df_new["Sentiment_Score"], errors="coerce").fillna(0.0)
        roll = s.rolling(window=252, min_periods=1)
        norm = ((s - roll.mean()) / (roll.std() + 1e-8)).clip(-3, 3) / 3
        frames.append(norm.rename("_s"))

        s_norm = pd.concat(frames).sort_index()
        s_norm = s_norm[~s_norm.index.duplicated(keep="last")]

        result = pd.DataFrame(index=s_norm.index)
        result["sentiment_score"]    = s_norm
        result["sentiment_std"]      = 0.0
        result["article_count"]      = 1.0
        result["sentiment_ma_3d"]    = s_norm.rolling(3,  min_periods=1).mean()
        result["sentiment_ma_5d"]    = s_norm.rolling(5,  min_periods=1).mean()
        result["sentiment_ma_10d"]   = s_norm.rolling(10, min_periods=1).mean()
        result["sentiment_momentum"] = result["sentiment_ma_3d"] - result["sentiment_ma_10d"]
        result["buzz_ma_3d"]         = 1.0
        result["buzz_ma_5d"]         = 1.0
        result["sentiment_std_5d"]   = s_norm.rolling(5, min_periods=1).std().fillna(0.0)
        return result.fillna(0.0)

    def _check_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found at {path}. "
                f"Copy it to ml/data/raw/sentiment/"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Nifty 50 feature matrix")
    parser.add_argument(
        "--finbert", action="store_true",
        help="Use FinBERT for sentiment scoring (slow on CPU, requires GPU ideally)"
    )
    args   = parser.parse_args()
    loader = NiftyLoader()
    df     = loader.build_feature_matrix(use_finbert=args.finbert)
    print(f"\nNifty 50 feature matrix:")
    print(f"  Rows:     {df.shape[0]}")
    print(f"  Cols:     {df.shape[1]}")
    print(f"  Dates:    {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Target:   log_return_5d  (5-day forward log return)")
    print(f"\nData coverage by year:")
    for year in range(2008, 2025):
        try:
            n = len(df.loc[str(year)])
            if n > 0:
                print(f"  {year}: {n:3d} rows")
        except KeyError:
            pass