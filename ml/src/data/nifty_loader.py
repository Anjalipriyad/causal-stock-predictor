"""
nifty_loader.py
---------------
Loads and preprocesses the three uploaded Indian market data files:

    1. Final_Data.csv
       Nifty 50 OHLCV + P/E + P/B + India VIX + Sentiment_Score
       2019-01-01 to 2024-11-01 (1,357 rows)
       Source: NSE India + Economic Times (VADER sentiment)

    2. financial_headlines_with_serial_no.csv
       40,044 Indian financial headlines (2019-2023)

    3. financial_headlines_with_serial_no_2.csv
       28,995 Indian financial headlines (2009-2018)

Produces a feature matrix compatible with the existing PCMCI + ensemble
pipeline. Also computes Trend and Future_Trend labels matching the
StockAI 3.0 paper (Springer) for direct baseline comparison.

Store files at:
    ml/data/raw/sentiment/Final_Data.csv
    ml/data/raw/sentiment/financial_headlines_with_serial_no.csv
    ml/data/raw/sentiment/financial_headlines_with_serial_no_2.csv

Usage:
    from ml.src.data.nifty_loader import NiftyLoader
    loader = NiftyLoader()
    df     = loader.build_feature_matrix()

    # Or from CLI
    python -m ml.src.data.nifty_loader
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
    Loads Indian market data from the three uploaded files and builds
    a feature matrix compatible with the existing FeaturePipeline.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg      = _load_config(config_path)
        self.root     = Path(__file__).resolve().parents[3]
        self.data_dir = self.root / self.cfg["data"]["raw_dir"] / "sentiment"

        self.final_data_path = self.data_dir / "Final_Data.csv"
        self.headlines_path1 = self.data_dir / "financial_headlines_with_serial_no.csv"
        self.headlines_path2 = self.data_dir / "financial_headlines_with_serial_no_2.csv"

        self.out_dir = self.root / self.cfg["data"]["processed_dir"] / "features"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def load_prices(self) -> pd.DataFrame:
        """
        Load Nifty 50 OHLCV from Final_Data.csv.
        Returns DataFrame with DatetimeIndex matching existing pipeline format.
        Columns: open, high, low, close, volume
        """
        self._check_file(self.final_data_path)
        df = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        price_df = pd.DataFrame(index=df.index)
        price_df["open"]   = df["Open"]
        price_df["close"]  = df["Close"]
        price_df["volume"] = df["Volume"]
        # Approximate high/low (not in source)
        price_df["high"] = df[["Open", "Close"]].max(axis=1) * 1.005
        price_df["low"]  = df[["Open", "Close"]].min(axis=1) * 0.995

        logger.info(
            f"[nifty_loader] Prices: {len(price_df)} rows "
            f"({price_df.index.min().date()} → {price_df.index.max().date()})"
        )
        return price_df

    def load_fundamental_features(self) -> pd.DataFrame:
        """
        Load P/E, P/B, India VIX features from Final_Data.csv.
        These are additional causal candidates not available in US pipeline.
        """
        self._check_file(self.final_data_path)
        df = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        result = pd.DataFrame(index=df.index)

        # Raw values
        result["pe_ratio"]  = df["P/E"]
        result["pb_ratio"]  = df["P/B"]
        result["india_vix"] = df["Vix"]

        # Changes
        result["pe_change_1d"]  = df["P/E"].diff(1)
        result["pb_change_1d"]  = df["P/B"].diff(1)
        result["vix_change_1d"] = df["Vix"].diff(1)
        result["vix_change_5d"] = df["Vix"].diff(5)

        # Moving averages
        result["vix_ma_5"]  = df["Vix"].rolling(5).mean()
        result["vix_ma_10"] = df["Vix"].rolling(10).mean()
        result["pe_ma_10"]  = df["P/E"].rolling(10).mean()
        result["pb_ma_10"]  = df["P/B"].rolling(10).mean()

        # Percentile rank — where is today vs last year
        result["vix_percentile"] = df["Vix"].rolling(252).rank(pct=True)
        result["pe_percentile"]  = df["P/E"].rolling(252).rank(pct=True)
        result["pb_percentile"]  = df["P/B"].rolling(252).rank(pct=True)

        # Valuation regime: expensive vs cheap
        result["pe_regime"]      = (df["P/E"] > df["P/E"].rolling(60).mean()).astype(float)
        result["vix_regime"]     = (df["Vix"] > df["Vix"].rolling(60).mean()).astype(float)

        # PE × VIX interaction — high PE + high VIX = danger signal
        result["pe_vix_danger"]  = result["pe_percentile"] * result["vix_percentile"]

        logger.info(f"[nifty_loader] Fundamental features: {len(result.columns)} features")
        return result.fillna(0)

    def load_sentiment(self, use_finbert: bool = False) -> pd.DataFrame:
        """
        Load and score Indian financial headlines.
        Combines both headline files.
        Falls back to pre-computed VADER scores from Final_Data.csv if no headlines.

        Args:
            use_finbert: Score with FinBERT (slow, needs GPU). Default False = keyword scoring.

        Returns DataFrame with daily sentiment features.
        """
        headlines_df = self._load_all_headlines()

        if headlines_df.empty:
            logger.warning("[nifty_loader] No headlines found — using pre-computed VADER scores.")
            return self._load_precomputed_sentiment()

        # Score headlines
        if use_finbert:
            logger.info("[nifty_loader] Scoring with FinBERT...")
            from ml.src.features.finbert import FinBERTSentiment
            scorer = FinBERTSentiment()
            if scorer.is_available():
                scores = scorer.score_batch(headlines_df["headline"].tolist())
            else:
                logger.warning("[nifty_loader] FinBERT not available — using keyword scoring.")
                scores = [scorer._keyword_score(h) for h in headlines_df["headline"]]
        else:
            from ml.src.features.finbert import FinBERTSentiment
            scorer = FinBERTSentiment()
            scores = [scorer._keyword_score(h) for h in headlines_df["headline"]]

        headlines_df["score"] = scores

        # Aggregate to daily
        daily = headlines_df.groupby("date")["score"].agg(["mean", "std", "count"])
        daily.columns = ["sentiment_score", "sentiment_std", "article_count"]
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
            f"{int(daily['article_count'].sum())} total headlines scored"
        )
        return daily.fillna(0)

    def compute_labels(self) -> pd.DataFrame:
        """
        Compute Trend and Future_Trend labels matching StockAI 3.0 paper.
        Used for direct baseline comparison.

        Trend:        Close > Open → Positive, else Negative
        Future_Trend: Close[t+7] > Close[t] → Bullish, else Bearish
        """
        self._check_file(self.final_data_path)
        df = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

        labels = pd.DataFrame(index=df.index)
        labels["trend"]        = (df["Close"] > df["Open"]).astype(int)
        labels["future_trend"] = (df["Close"].shift(-7) > df["Close"]).astype(int)

        logger.info(
            f"[nifty_loader] Labels computed. "
            f"Bullish days: {labels['future_trend'].sum()} / {len(labels)}"
        )
        return labels

    def build_feature_matrix(
        self,
        use_finbert: bool = False,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build complete Nifty 50 feature matrix ready for PCMCI + model training.

        Combines:
            Technical features     (RSI, MACD, Bollinger, momentum, ATR)
            Fundamental features   (P/E, P/B, India VIX — unique to this dataset)
            Sentiment features     (from 69,000 Economic Times headlines)
            Target variable        (excess_return_5d vs Nifty itself)

        Returns DataFrame saved to ml/data/processed/features/NIFTY_features.csv
        """
        from ml.src.features.technical import TechnicalFeatures

        logger.info("[nifty_loader] Building Nifty 50 feature matrix...")

        # 1. Prices
        price_df = self.load_prices()

        # 2. Technical features (same as US pipeline)
        tech = TechnicalFeatures()
        # For Nifty index — no benchmark (predicting absolute return)
        # Do NOT set spy_close to itself — that makes excess_return_5d = 0
        tech.set_spy_close(None)
        tech_df = tech.compute(price_df)

        # Rename target: for index prediction use log_return_5d not excess_return_5d
        # excess_return vs itself is always 0 — meaningless
        if "excess_return_5d" in tech_df.columns:
            tech_df = tech_df.rename(columns={"excess_return_5d": "log_return_5d_target"})
            tech_df["excess_return_5d"] = tech_df["log_return_5d_target"]
            tech_df = tech_df.drop(columns=["log_return_5d_target"], errors="ignore")

        # 3. Fundamental features (unique to this dataset)
        fund_df = self.load_fundamental_features()
        fund_df = fund_df.reindex(tech_df.index).ffill().fillna(0)

        # 4. Sentiment from headlines
        sent_df = self.load_sentiment(use_finbert=use_finbert)
        sent_df.index = pd.to_datetime(sent_df.index)
        sent_df = sent_df.reindex(tech_df.index).ffill(limit=5).fillna(0)

        # 5. Merge all
        df = tech_df.copy()
        df = df.join(fund_df, how="left")
        df = df.join(sent_df, how="left")

        # 6. For NIFTY: keep log_return_5d as it is the prediction target
        # For individual stocks: log_return_5d would leak into excess_return_5d target
        # Since we renamed excess_return_5d to use raw return above, keep log_return_5d
        pass  # do not drop log_return_5d for NIFTY

        # 7. Final fill
        df = df.ffill(limit=5).fillna(0)

        logger.info(
            f"[nifty_loader] Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols"
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
        """Load and combine both headline files."""
        frames = []
        for path in [self.headlines_path1, self.headlines_path2]:
            if not path.exists():
                logger.warning(f"[nifty_loader] Not found: {path.name}")
                continue
            try:
                df = pd.read_csv(path)[["date", "headline"]].dropna()
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
        """Normalise pre-computed VADER scores from Final_Data.csv."""
        df = pd.read_csv(self.final_data_path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()

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
    print(f"  Columns:  {df.columns.tolist()}")