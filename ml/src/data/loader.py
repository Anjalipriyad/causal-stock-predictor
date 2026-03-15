"""
loader.py
---------
Two responsibilities:
  1. load_historical(ticker) — bulk pull 2010-2025, saves to data/raw/
                               run ONCE before training, never again
  2. load_live(ticker)       — pulls last 90 days + latest sentiment
                               run at every inference call
                               returns a single clean DataFrame row

Usage:
    from ml.src.data.loader import DataLoader
    loader = DataLoader()

    # Once before training
    loader.load_historical("AAPL")

    # At every inference call
    live_df = loader.load_live("AAPL")
"""

import os
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml, substituting ${ENV_VAR} placeholders from environment."""
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
        )
    with open(config_path, "r") as f:
        raw = f.read()
    for match in re.finditer(r"\$\{(\w+)\}", raw):
        env_key = match.group(1)
        env_val = os.getenv(env_key, "")
        raw = raw.replace(match.group(0), env_val)
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Loads raw price, macro, and sentiment data from yFinance and Finnhub.

    Directory layout written after loading:
        data/raw/prices/        {ticker}.csv
        data/raw/macro/         {safe_symbol}.csv
        data/raw/sentiment/     {ticker}_sentiment.csv
        data/live/              {ticker}_live_prices.csv
                                {ticker}_live_sentiment.json
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg  = _load_config(config_path)
        self.root = Path(__file__).resolve().parents[3]   # project root

        # Date range from config
        self.start = self.cfg["data"]["start_date"]
        self.end   = self.cfg["data"]["end_date"]

        # Directories
        raw  = self.cfg["data"]["raw_dir"]
        live = self.cfg["data"]["live_dir"]
        self.prices_dir    = self.root / raw  / "prices"
        self.macro_dir     = self.root / raw  / "macro"
        self.sentiment_dir = self.root / raw  / "sentiment"
        self.live_dir      = self.root / live

        for d in [self.prices_dir, self.macro_dir,
                  self.sentiment_dir, self.live_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Finnhub config
        self.finnhub_key    = self.cfg["finnhub"]["api_key"]
        self.finnhub_base   = self.cfg["finnhub"]["base_url"]
        self.request_delay  = self.cfg["finnhub"]["request_delay_seconds"]
        self.max_retries    = self.cfg["finnhub"]["max_retries"]
        self.timeout        = self.cfg["finnhub"]["timeout_seconds"]
        self.sentiment_window = self.cfg["finnhub"]["sentiment_window_days"]

        # Live lookback
        self.live_lookback  = self.cfg["data"]["live_lookback_days"]

        # Tickers
        self.macro_tickers  = (
            self.cfg["data"]["tickers"]["macro"]
            + self.cfg["data"]["tickers"]["sector_etfs"]
        )

    # -----------------------------------------------------------------------
    # PUBLIC — Historical (run once)
    # -----------------------------------------------------------------------

    def load_historical(self, ticker: str) -> None:
        """
        Full historical pipeline for one target ticker.
        Pulls prices + all macro + sentiment from 2010 to 2025.
        Skips any file that already exists on disk.
        """
        ticker = ticker.upper()
        logger.info(f"=== Historical load: {ticker} ===")
        self.load_price(ticker, self.start, self.end, dest=self.prices_dir)
        self.load_macro_historical()
        self.load_sentiment_historical(ticker)
        logger.info(f"=== Done: {ticker} ===")

    def load_price(
        self,
        ticker: str,
        start: str,
        end: str,
        dest: Path,
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Pull OHLCV from yFinance for any ticker and date range.
        Saves to dest/{filename or ticker}.csv
        Returns the DataFrame.
        """
        fname    = filename or f"{ticker}.csv"
        out_path = dest / fname

        if out_path.exists():
            logger.info(f"[price] {ticker}: already on disk, loading.")
            return pd.read_csv(out_path, index_col=0, parse_dates=True)

        logger.info(f"[price] Downloading {ticker} {start} → {end} ...")
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        if df.empty:
            raise ValueError(f"yFinance returned empty data for {ticker}")

        df.index.name = "date"
        df.columns    = [c.lower() for c in df.columns]

        # Keep only configured columns
        keep = self.cfg["data"]["price_columns"]
        df   = df[[c for c in keep if c in df.columns]]

        df.to_csv(out_path)
        logger.info(f"[price] Saved {len(df)} rows → {out_path.name}")
        return df

    def load_macro_historical(self) -> dict[str, pd.DataFrame]:
        """
        Pull all macro + sector ETF tickers for the full date range.
        One CSV per symbol saved to data/raw/macro/
        """
        results = {}
        for symbol in self.macro_tickers:
            safe   = self._safe_filename(symbol)
            out_path = self.macro_dir / f"{safe}.csv"

            if out_path.exists():
                logger.info(f"[macro] {symbol}: already on disk, skipping.")
                results[symbol] = pd.read_csv(
                    out_path, index_col=0, parse_dates=True
                )
                continue

            logger.info(f"[macro] Downloading {symbol} ...")
            try:
                df = yf.download(
                    symbol,
                    start=self.start,
                    end=self.end,
                    auto_adjust=True,
                    progress=False,
                    multi_level_index=False,
                )
                if df.empty:
                    logger.warning(f"[macro] Empty data for {symbol}, skipping.")
                    continue

                df.index.name = "date"
                df.columns    = [c.lower() for c in df.columns]
                df.to_csv(out_path)
                logger.info(f"[macro] Saved {len(df)} rows → {out_path.name}")
                results[symbol] = df
                time.sleep(0.3)

            except Exception as e:
                logger.warning(f"[macro] Failed {symbol}: {e}")

        return results

    def load_sentiment_historical(self, ticker: str) -> pd.DataFrame:
        """
        Pull daily news sentiment from Finnhub for the full date range.
        Iterates in 30-day windows (free tier limit).
        Saves to data/raw/sentiment/{ticker}_sentiment.csv

        Columns saved:
            date, article_count, avg_sentiment
        """
        out_path = self.sentiment_dir / f"{ticker}_sentiment.csv"
        if out_path.exists():
            logger.info(f"[sentiment] {ticker}: already on disk, skipping.")
            return pd.read_csv(out_path, index_col=0, parse_dates=True)

        if not self.finnhub_key:
            logger.warning("[sentiment] FINNHUB_API_KEY not set — skipping.")
            return pd.DataFrame()

        logger.info(f"[sentiment] Fetching {ticker} sentiment {self.start} → {self.end} ...")

        records    = []
        current    = datetime.strptime(self.start, "%Y-%m-%d")
        end_dt     = datetime.strptime(self.end,   "%Y-%m-%d")

        while current < end_dt:
            window_end = min(
                current + timedelta(days=self.sentiment_window), end_dt
            )
            data = self._finnhub_get(
                "/company-news",
                {
                    "symbol": ticker,
                    "from":   current.strftime("%Y-%m-%d"),
                    "to":     window_end.strftime("%Y-%m-%d"),
                },
            )
            if data:
                daily = self._aggregate_sentiment(data, current, window_end)
                records.extend(daily)

            time.sleep(self.request_delay)
            current = window_end + timedelta(days=1)

        if not records:
            logger.warning(f"[sentiment] No data for {ticker}.")
            return pd.DataFrame()

        df = (
            pd.DataFrame(records)
            .set_index("date")
            .sort_index()
        )
        df.index = pd.to_datetime(df.index)
        df.to_csv(out_path)
        logger.info(f"[sentiment] Saved {len(df)} rows → {out_path.name}")
        return df

    # -----------------------------------------------------------------------
    # PUBLIC — Live (run at every inference call)
    # -----------------------------------------------------------------------

    def load_live(self, ticker: str) -> dict[str, pd.DataFrame]:
        """
        Pull the last `live_lookback_days` of data for inference.
        Overwrites data/live/ each call — not appended, always fresh.

        Returns:
            {
                "prices":    DataFrame  — OHLCV for last 90 days
                "macro":     DataFrame  — macro close prices for last 90 days
                "sentiment": DataFrame  — sentiment for last 90 days
            }
        """
        ticker   = ticker.upper()
        end_dt   = datetime.today()
        start_dt = end_dt - timedelta(days=self.live_lookback)
        start    = start_dt.strftime("%Y-%m-%d")
        end      = end_dt.strftime("%Y-%m-%d")

        logger.info(f"[live] Fetching {ticker} {start} → {end}")

        # Prices
        prices = self.load_price(
            ticker, start, end,
            dest=self.live_dir,
            filename=f"{ticker}_live_prices.csv",
        )

        # Macro — overwrite each time for live
        macro_frames = {}
        for symbol in self.macro_tickers:
            safe = self._safe_filename(symbol)
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    multi_level_index=False,
                )
                if not df.empty:
                    df.index.name = "date"
                    df.columns    = [c.lower() for c in df.columns]
                    # Keep symbol as column name so _split_macro_wide can reverse-map it
                    macro_frames[symbol] = df[["close"]].rename(
                        columns={"close": symbol}
                    )
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"[live macro] {symbol}: {e}")

        macro_df = pd.concat(macro_frames.values(), axis=1) if macro_frames else pd.DataFrame()

        # Sentiment — last 30 days only (Finnhub free tier)
        sentiment_start = (end_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        sentiment = self._fetch_sentiment_window(ticker, sentiment_start, end)

        return {
            "prices":    prices,
            "macro":     macro_df,
            "sentiment": sentiment,
        }

    # -----------------------------------------------------------------------
    # PRIVATE — Finnhub helpers
    # -----------------------------------------------------------------------

    def _finnhub_get(self, path: str, params: dict) -> Optional[list | dict]:
        """GET request to Finnhub with retry logic."""
        if not self.finnhub_key:
            return None
        params["token"] = self.finnhub_key
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    f"{self.finnhub_base}{path}",
                    params=params,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"[finnhub] Rate limited, waiting {wait}s ...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                logger.warning(f"[finnhub] Attempt {attempt + 1} failed: {e}")
                time.sleep(self.request_delay)
        return None

    def _fetch_sentiment_window(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch and aggregate sentiment for a single date window."""
        data = self._finnhub_get(
            "/company-news",
            {"symbol": ticker, "from": start, "to": end},
        )
        if not data:
            return pd.DataFrame()
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")
        records  = self._aggregate_sentiment(data, start_dt, end_dt)
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).set_index("date")
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _aggregate_sentiment(
        self,
        news_items: list,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """
        Group Finnhub news items by date and compute:
            - article_count   : number of articles that day
            - avg_sentiment   : mean sentiment score (-1 to +1)
                                derived from headline keyword scoring
                                (Finnhub free tier has no per-article score)
        """
        from collections import defaultdict

        # Simple lexicon-based sentiment proxy for free tier
        POSITIVE = {
            "beat", "beats", "surge", "surges", "rises", "gain", "gains",
            "profit", "record", "growth", "upgrade", "strong", "bullish",
            "rally", "outperform", "exceed", "exceeds", "positive", "up",
        }
        NEGATIVE = {
            "miss", "misses", "fall", "falls", "drop", "drops", "loss",
            "losses", "decline", "declines", "downgrade", "weak", "bearish",
            "crash", "slump", "underperform", "below", "negative", "down",
            "risk", "risks", "cut", "cuts",
        }

        daily: dict = defaultdict(lambda: {"scores": [], "count": 0})

        for item in news_items:
            try:
                ts       = item.get("datetime", 0)
                headline = item.get("headline", "").lower()
                date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

                # Score headline
                words  = set(headline.split())
                pos    = len(words & POSITIVE)
                neg    = len(words & NEGATIVE)
                total  = pos + neg
                score  = (pos - neg) / total if total > 0 else 0.0

                daily[date_str]["scores"].append(score)
                daily[date_str]["count"] += 1
            except Exception:
                continue

        records = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            entry    = daily.get(date_str)
            records.append({
                "date":          date_str,
                "article_count": entry["count"] if entry else 0,
                "avg_sentiment": float(np.mean(entry["scores"]))
                                 if entry and entry["scores"] else 0.0,
            })
            current += timedelta(days=1)

        return records

    # -----------------------------------------------------------------------
    # PRIVATE — Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_filename(symbol: str) -> str:
        """Convert ticker symbol to a safe filename string."""
        return (
            symbol
            .replace("^", "")
            .replace("=", "_")
            .replace("-", "_")
            .replace(".", "_")
        )

    # -----------------------------------------------------------------------
    # PUBLIC — Convenience readers (used by feature pipeline)
    # -----------------------------------------------------------------------

    def read_prices(self, ticker: str) -> pd.DataFrame:
        """Read saved historical prices from disk."""
        path = self.prices_dir / f"{ticker.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No price data for {ticker}. Run load_historical('{ticker}') first."
            )
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def read_macro(self, symbol: str) -> pd.DataFrame:
        """Read saved historical macro data from disk."""
        safe = self._safe_filename(symbol)
        path = self.macro_dir / f"{safe}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No macro data for {symbol}. Run load_historical() first."
            )
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def read_sentiment(self, ticker: str) -> pd.DataFrame:
        """Read saved historical sentiment from disk."""
        path = self.sentiment_dir / f"{ticker.upper()}_sentiment.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No sentiment data for {ticker}. Run load_historical('{ticker}') first."
            )
        return pd.read_csv(path, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load historical data for a ticker")
    parser.add_argument("--ticker", type=str, required=True, help="e.g. AAPL")
    parser.add_argument("--live",   action="store_true", help="Load live data instead")
    args = parser.parse_args()

    loader = DataLoader()
    if args.live:
        result = loader.load_live(args.ticker)
        print(f"Live data loaded: {list(result.keys())}")
    else:
        loader.load_historical(args.ticker)
        print(f"Historical data loaded for {args.ticker}")