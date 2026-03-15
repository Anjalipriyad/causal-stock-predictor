"""
test_loader.py
--------------
Tests for DataLoader — data fetching + validation.
Uses mocked yFinance and Finnhub responses so tests run offline.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd
import numpy as np

from ml.src.data.loader import DataLoader, _load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader(tmp_path):
    """DataLoader with all directories redirected to tmp_path."""
    loader = DataLoader()
    loader.prices_dir    = tmp_path / "raw" / "prices"
    loader.macro_dir     = tmp_path / "raw" / "macro"
    loader.sentiment_dir = tmp_path / "raw" / "sentiment"
    loader.live_dir      = tmp_path / "live"
    for d in [loader.prices_dir, loader.macro_dir,
              loader.sentiment_dir, loader.live_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return loader


@pytest.fixture
def sample_price_df():
    """Minimal OHLCV DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, 100).astype(float),
    }, index=pd.DatetimeIndex(dates, name="date"))


@pytest.fixture
def sample_news_items():
    """Sample Finnhub news items."""
    base_ts = int(datetime(2020, 1, 2).timestamp())
    return [
        {"datetime": base_ts,       "headline": "Apple beats earnings expectations"},
        {"datetime": base_ts,       "headline": "Apple stock surges on record profit"},
        {"datetime": base_ts + 86400, "headline": "Apple faces regulatory risk decline"},
    ]


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self):
        cfg = _load_config()
        assert isinstance(cfg, dict)

    def test_has_required_keys(self):
        cfg = _load_config()
        assert "data" in cfg
        assert "finnhub" in cfg
        assert "model" in cfg
        assert "causal" in cfg
        assert "evaluation" in cfg

    def test_date_range(self):
        cfg = _load_config()
        assert cfg["data"]["start_date"] == "2010-01-01"
        assert cfg["data"]["end_date"]   == "2025-12-31"

    def test_env_var_substitution(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test_key_123")
        cfg = _load_config()
        assert cfg["finnhub"]["api_key"] == "test_key_123"


# ---------------------------------------------------------------------------
# DataLoader.load_price
# ---------------------------------------------------------------------------

class TestLoadPrice:
    def test_saves_csv(self, loader, sample_price_df):
        with patch("yfinance.download", return_value=sample_price_df):
            result = loader.load_price("AAPL", "2020-01-01", "2020-06-01",
                                       dest=loader.prices_dir)
        out = loader.prices_dir / "AAPL.csv"
        assert out.exists()
        assert len(result) == 100

    def test_skips_existing(self, loader, sample_price_df):
        """Should not call yfinance if file already exists."""
        out = loader.prices_dir / "AAPL.csv"
        sample_price_df.to_csv(out)
        with patch("yfinance.download") as mock_dl:
            loader.load_price("AAPL", "2020-01-01", "2020-06-01",
                              dest=loader.prices_dir)
        mock_dl.assert_not_called()

    def test_raises_on_empty_response(self, loader):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="empty"):
                loader.load_price("FAKE", "2020-01-01", "2020-06-01",
                                  dest=loader.prices_dir)

    def test_columns_lowercased(self, loader, sample_price_df):
        with patch("yfinance.download", return_value=sample_price_df):
            result = loader.load_price("AAPL", "2020-01-01", "2020-06-01",
                                       dest=loader.prices_dir)
        assert all(c == c.lower() for c in result.columns)

    def test_index_named_date(self, loader, sample_price_df):
        with patch("yfinance.download", return_value=sample_price_df):
            result = loader.load_price("AAPL", "2020-01-01", "2020-06-01",
                                       dest=loader.prices_dir)
        assert result.index.name == "date"


# ---------------------------------------------------------------------------
# DataLoader._aggregate_sentiment
# ---------------------------------------------------------------------------

class TestAggregateSentiment:
    def test_returns_list_of_dicts(self, loader, sample_news_items):
        start = datetime(2020, 1, 1)
        end   = datetime(2020, 1, 3)
        result = loader._aggregate_sentiment(sample_news_items, start, end)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_keys_present(self, loader, sample_news_items):
        start  = datetime(2020, 1, 1)
        end    = datetime(2020, 1, 3)
        result = loader._aggregate_sentiment(sample_news_items, start, end)
        for row in result:
            assert "date" in row
            assert "article_count" in row
            assert "avg_sentiment" in row

    def test_sentiment_in_range(self, loader, sample_news_items):
        start  = datetime(2020, 1, 1)
        end    = datetime(2020, 1, 3)
        result = loader._aggregate_sentiment(sample_news_items, start, end)
        for row in result:
            assert -1.0 <= row["avg_sentiment"] <= 1.0

    def test_article_count_correct(self, loader, sample_news_items):
        start  = datetime(2020, 1, 2)
        end    = datetime(2020, 1, 2)
        result = loader._aggregate_sentiment(sample_news_items, start, end)
        day2   = next(r for r in result if r["date"] == "2020-01-02")
        assert day2["article_count"] == 2

    def test_empty_news_returns_zeros(self, loader):
        start  = datetime(2020, 1, 1)
        end    = datetime(2020, 1, 3)
        result = loader._aggregate_sentiment([], start, end)
        for row in result:
            assert row["article_count"] == 0
            assert row["avg_sentiment"] == 0.0


# ---------------------------------------------------------------------------
# DataLoader.read_prices / read_macro / read_sentiment
# ---------------------------------------------------------------------------

class TestReaders:
    def test_read_prices_raises_if_missing(self, loader):
        with pytest.raises(FileNotFoundError, match="AAPL"):
            loader.read_prices("AAPL")

    def test_read_prices_returns_df(self, loader, sample_price_df):
        sample_price_df.to_csv(loader.prices_dir / "AAPL.csv")
        result = loader.read_prices("AAPL")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_read_macro_raises_if_missing(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.read_macro("^VIX")

    def test_read_sentiment_raises_if_missing(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.read_sentiment("AAPL")


# ---------------------------------------------------------------------------
# DataLoader._safe_filename
# ---------------------------------------------------------------------------

class TestSafeFilename:
    @pytest.mark.parametrize("symbol,expected", [
        ("^VIX",     "VIX"),
        ("GC=F",     "GC_F"),
        ("DX-Y.NYB", "DX_Y_NYB"),
        ("XLK",      "XLK"),
    ])
    def test_safe_filename(self, symbol, expected):
        result = DataLoader._safe_filename(symbol)
        assert result == expected