"""
nifty_loader.py
---------------
Loads and preprocesses all Indian market data files:

    Price + Fundamental data:
        Unused_Data.csv       — 2008-01-28 to 2018-12-27
                                OHLC + P/E + P/B + VIX + Sentiment + Trend
                                (no Volume column)
        Final_Data.csv        — 2019-01-01 to 2024-11-01
                                OHLCV + P/E + P/B + VIX + Sentiment

    Headlines (sentiment):
        financial_headlines_with_serial_no_2.csv   — 2008-2018 (28,995 headlines)
        financial_headlines_with_serial_no.csv     — 2019-2024 (40,044 headlines)

Combined coverage: 2008-2024 (~3,555 rows, 69,000+ headlines)

Regimes covered:
    Global Financial Crisis    2008-2009
    GFC Recovery               2009-2013
    Pre-demonetisation bull    2014-2016
    Demonetisation shock       Nov 2016 - Mar 2017
    Post-demo recovery         2017-2019
    COVID crash                Jan-Jun 2020
    COVID recovery             Jul 2020-2021
    RBI rate hike              2022-2023
    Current bull               2023-2024

Store files at: ml/data/raw/sentiment/

Usage:
    from ml.src.data.nifty_loader import NiftyLoader
    loader = NiftyLoader()
    df     = loader.build_feature_matrix()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

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

        # Price + fundamental files
        self.final_data_path  = self.data_dir / "Final_Data.csv"
        self.unused_data_path = self.data_dir / "Unused_Data.csv"

        # Headline files
        self.headlines_old    = self.data_dir / "financial_headlines_with_serial_no_2.csv"
        self.headlines_new    = self.data_dir / "financial_headlines_with_serial_no.csv"

        self.out_dir = self.root / self.cfg["data"]["processed_dir"] / "features"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        """
        Load Nifty 50 OHLCV from both CSVs combined (2008-2024).
        Unused_Data has no Volume — fills with 0 for that period.
        Returns DataFrame with DatetimeIndex.
        Columns: open, high, low, close, volume
        """
        frames = []

        # 2008-2018 from Unused_Data.csv
        if self.unused_data_path.exists():
            df_old = pd.read_csv(self.unused_data_path, parse_dates=["Date"])
            df_old = df_old.set_index("Date").sort_index()
            old_prices = pd.DataFrame(index=df_old.index)
            old_prices["open"]   = df_old["Open"]
            old_prices["close"]  = df_old["Close"]
            old_prices["volume"] = 0   # not available for this period
            old_prices["high"]   = df_old[["Open","Close"]].max(axis=1) * 1.005
            old_prices["low"]    = df_old[["Open","Close"]].min(axis=1) * 0.995
            frames.append(old_prices)
            logger.info(f"[nifty_loader] Unused_Data: {len(df_old)} rows "
                       f"({df_old.index.min().date()} → {df_old.index.max().date()})")

        # 2019-2024 from Final_Data.csv
        self._check_file(self.final_data_path)
        df_new = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df_new = df_new.set_index("Date").sort_index()
        new_prices = pd.DataFrame(index=df_new.index)
        new_prices["open"]   = df_new["Open"]
        new_prices["close"]  = df_new["Close"]
        new_prices["volume"] = df_new["Volume"]
        new_prices["high"]   = df_new[["Open","Close"]].max(axis=1) * 1.005
        new_prices["low"]    = df_new[["Open","Close"]].min(axis=1) * 0.995
        frames.append(new_prices)
        logger.info(f"[nifty_loader] Final_Data: {len(df_new)} rows "
                   f"({df_new.index.min().date()} → {df_new.index.max().date()})")

        # Combine
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
        Computes regime signals, valuation features, VIX features.
        """
        frames = []

        if self.unused_data_path.exists():
            df_old = pd.read_csv(self.unused_data_path, parse_dates=["Date"])
            df_old = df_old.set_index("Date").sort_index()
            frames.append(df_old)

        self._check_file(self.final_data_path)
        df_new = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df_new = df_new.set_index("Date").sort_index()
        frames.append(df_new)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        result = pd.DataFrame(index=df.index)

        # Raw values
        result["pe_ratio"]  = df["P/E"]
        result["pb_ratio"]  = df["P/B"]
        result["india_vix"] = df["Vix"]

        # 1-day and 5-day changes
        result["pe_change_1d"]  = df["P/E"].diff(1)
        result["pb_change_1d"]  = df["P/B"].diff(1)
        result["vix_change_1d"] = df["Vix"].diff(1)
        result["vix_change_5d"] = df["Vix"].diff(5)

        # Moving averages
        result["vix_ma_5"]  = df["Vix"].rolling(5).mean()
        result["vix_ma_10"] = df["Vix"].rolling(10).mean()
        result["pe_ma_10"]  = df["P/E"].rolling(10).mean()
        result["pb_ma_10"]  = df["P/B"].rolling(10).mean()

        # Percentile ranks (where is today vs last year)
        result["vix_percentile"] = df["Vix"].rolling(252).rank(pct=True)
        result["pe_percentile"]  = df["P/E"].rolling(252).rank(pct=True)
        result["pb_percentile"]  = df["P/B"].rolling(252).rank(pct=True)

        # Regime signals
        result["pe_regime"]      = (df["P/E"] > df["P/E"].rolling(60).mean()).astype(float)
        result["vix_regime"]     = (df["Vix"] > df["Vix"].rolling(60).mean()).astype(float)

        # Danger signal: high PE + high VIX simultaneously
        result["pe_vix_danger"]  = result["pe_percentile"] * result["vix_percentile"]

        # Crisis detector: VIX > 30 is extreme fear
        result["vix_crisis"]     = (df["Vix"] > 30).astype(float)

        # Valuation extremes: market expensive (PE > 25) or cheap (PE < 15)
        result["market_expensive"] = (df["P/E"] > 25).astype(float)
        result["market_cheap"]     = (df["P/E"] < 15).astype(float)

        logger.info(f"[nifty_loader] Fundamental features: {len(result.columns)} features "
                   f"({result.index.min().date()} → {result.index.max().date()})")
        return result.fillna(0)

    def load_sentiment(self, use_finbert: bool = False) -> pd.DataFrame:
        """
        Load and score all Indian financial headlines (2008-2024).
        Combines both headline files.
        Falls back to pre-computed VADER scores if no headline files found.
        """
        headlines_df = self._load_all_headlines()

        if headlines_df.empty:
            logger.warning("[nifty_loader] No headlines found — using pre-computed VADER scores.")
            return self._load_precomputed_sentiment()

        # Score headlines
        from ml.src.features.finbert import FinBERTSentiment
        scorer = FinBERTSentiment()

        if use_finbert and scorer.is_available():
            logger.info("[nifty_loader] Scoring with FinBERT...")
            scores = scorer.score_batch(headlines_df["headline"].tolist())
        else:
            logger.info("[nifty_loader] Scoring with keyword model...")
            scores = [scorer._keyword_score(h) for h in headlines_df["headline"]]

        headlines_df["score"] = scores

        # Aggregate to daily
        daily = headlines_df.groupby("date")["score"].agg(["mean","std","count"])
        daily.columns = ["sentiment_score","sentiment_std","article_count"]
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)

        # Rolling features
        daily["sentiment_ma_3d"]    = daily["sentiment_score"].rolling(3).mean()
        daily["sentiment_ma_5d"]    = daily["sentiment_score"].rolling(5).mean()
        daily["sentiment_ma_10d"]   = daily["sentiment_score"].rolling(10).mean()
        daily["sentiment_momentum"] = daily["sentiment_ma_3d"] - daily["sentiment_ma_10d"]
        daily["buzz_ma_3d"]         = daily["article_count"].rolling(3).mean()
        daily["buzz_ma_5d"]         = daily["article_count"].rolling(5).mean()
        daily["sentiment_std_5d"]   = daily["sentiment_score"].rolling(5).std().fillna(0)

        logger.info(
            f"[nifty_loader] Sentiment: {len(daily)} days, "
            f"{int(daily['article_count'].sum())} headlines scored "
            f"({daily.index.min().date()} → {daily.index.max().date()})"
        )
        return daily.fillna(0)

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

        logger.info("[nifty_loader] Building Nifty 50 feature matrix (2008-2024)...")

        # 1. Prices
        price_df = self.load_prices()

        # 2. Technical features
        tech = TechnicalFeatures()
        tech.set_spy_close(None)   # no benchmark for index prediction
        tech_df = tech.compute(price_df)

        # 3. Fundamental features (P/E, P/B, India VIX)
        fund_df = self.load_fundamental_features()
        fund_df = fund_df.reindex(tech_df.index).ffill().fillna(0)

        # 4. Sentiment
        sent_df = self.load_sentiment(use_finbert=use_finbert)
        sent_df.index = pd.to_datetime(sent_df.index)
        sent_df = sent_df.reindex(tech_df.index).ffill(limit=5).fillna(0)

        # 5. Merge
        df = tech_df.copy()
        df = df.join(fund_df, how="left")
        df = df.join(sent_df, how="left")

        # 6. For NIFTY: keep log_return_5d as the prediction target
        # Do NOT drop it — it IS the target column for index prediction
        # (excess_return_5d is dropped by technical.py since spy_close=None makes it 0)
        # Rename excess_return_5d to log_return_5d if needed
        if "excess_return_5d" in df.columns and "log_return_5d" not in df.columns:
            df = df.rename(columns={"excess_return_5d": "log_return_5d"})
        elif "excess_return_5d" in df.columns:
            df = df.drop(columns=["excess_return_5d"], errors="ignore")

        # 7. Fill remaining NaN
        df = df.ffill(limit=5).fillna(0)

        logger.info(
            f"[nifty_loader] Feature matrix complete: "
            f"{df.shape[0]} rows × {df.shape[1]} cols "
            f"({df.index.min().date()} → {df.index.max().date()})"
        )

        if save:
            out_path = self.out_dir / "NIFTY_features.csv"
            df.to_csv(out_path)
            logger.info(f"[nifty_loader] Saved → {out_path.name}")

        return df

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _load_all_headlines(self) -> pd.DataFrame:
        """Load and combine all headline files (2008-2024)."""
        frames = []

        for path in [self.headlines_old, self.headlines_new]:
            if not path.exists():
                logger.warning(f"[nifty_loader] Not found: {path.name}")
                continue
            try:
                df = pd.read_csv(path)[["date","headline"]].dropna()
                df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                df = df.dropna(subset=["date"])
                df["date"] = df["date"].dt.normalize()
                frames.append(df)
                logger.info(f"[nifty_loader] Loaded {len(df)} headlines from {path.name}")
            except Exception as e:
                logger.warning(f"[nifty_loader] Failed to load {path.name}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info(
            f"[nifty_loader] Total headlines: {len(combined)} "
            f"({combined['date'].min().date()} → {combined['date'].max().date()})"
        )
        return combined

    def _load_precomputed_sentiment(self) -> pd.DataFrame:
        """Normalise pre-computed VADER scores from both CSVs combined."""
        frames = []
        if self.unused_data_path.exists():
            frames.append(
                pd.read_csv(self.unused_data_path, parse_dates=["Date"]).set_index("Date")
            )
        frames.append(
            pd.read_csv(self.final_data_path, parse_dates=["Date"]).set_index("Date")
        )
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        s      = df["Sentiment_Score"]
        s_norm = ((s - s.mean()) / s.std()).clip(-3, 3) / 3

        result = pd.DataFrame(index=df.index)
        result["sentiment_score"]    = s_norm
        result["sentiment_std"]      = 0.0
        result["article_count"]      = 1.0
        result["sentiment_ma_3d"]    = s_norm.rolling(3).mean()
        result["sentiment_ma_5d"]    = s_norm.rolling(5).mean()
        result["sentiment_ma_10d"]   = s_norm.rolling(10).mean()
        result["sentiment_momentum"] = result["sentiment_ma_3d"] - result["sentiment_ma_10d"]
        result["buzz_ma_3d"]         = 1.0
        result["buzz_ma_5d"]         = 1.0
        result["sentiment_std_5d"]   = s_norm.rolling(5).std().fillna(0)
        return result.fillna(0)

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
    parser.add_argument("--finbert", action="store_true",
                        help="Use FinBERT for sentiment scoring")
    args   = parser.parse_args()
    loader = NiftyLoader()
    df     = loader.build_feature_matrix(use_finbert=args.finbert)
    print(f"\nNifty 50 feature matrix:")
    print(f"  Rows:     {df.shape[0]}")
    print(f"  Cols:     {df.shape[1]}")
    print(f"  Dates:    {df.index.min().date()} → {df.index.max().date()}")