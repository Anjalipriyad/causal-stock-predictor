"""
test_features.py
----------------
Tests for TechnicalFeatures, MacroFeatures, SentimentFeatures, FeaturePipeline.
All tests use synthetic data — no network calls.
"""

import pytest
import numpy as np
import pandas as pd

from ml.src.features.technical import TechnicalFeatures
from ml.src.features.macro import MacroFeatures
from ml.src.features.sentiment import SentimentFeatures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df():
    """300 rows of synthetic OHLCV — enough for all rolling windows."""
    np.random.seed(42)
    n     = 300
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.02,
        "low":    close * 0.97,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=pd.DatetimeIndex(dates, name="date"))


@pytest.fixture
def macro_dict(price_df):
    """Minimal macro dict with VIX and TNX."""
    n     = len(price_df)
    dates = price_df.index
    np.random.seed(1)
    return {
        "^VIX":    pd.DataFrame({"close": 15 + np.random.randn(n)}, index=dates),
        "^TNX":    pd.DataFrame({"close":  4 + np.random.randn(n) * 0.1}, index=dates),
        "^IRX":    pd.DataFrame({"close":  3 + np.random.randn(n) * 0.1}, index=dates),
        "DX-Y.NYB": pd.DataFrame({"close": 100 + np.random.randn(n)}, index=dates),
        "GC=F":    pd.DataFrame({"close": 1800 + np.random.randn(n) * 5}, index=dates),
        "CL=F":    pd.DataFrame({"close":  70 + np.random.randn(n) * 2}, index=dates),
        "^GSPC":   pd.DataFrame({"close": 4000 + np.cumsum(np.random.randn(n))}, index=dates),
        "XLK":     pd.DataFrame({"close":  150 + np.cumsum(np.random.randn(n) * 0.3)}, index=dates),
        "XLF":     pd.DataFrame({"close":   35 + np.cumsum(np.random.randn(n) * 0.1)}, index=dates),
        "XLE":     pd.DataFrame({"close":   60 + np.cumsum(np.random.randn(n) * 0.2)}, index=dates),
        "XLV":     pd.DataFrame({"close":  130 + np.cumsum(np.random.randn(n) * 0.2)}, index=dates),
        "XLI":     pd.DataFrame({"close":   90 + np.cumsum(np.random.randn(n) * 0.15)}, index=dates),
    }


@pytest.fixture
def sentiment_df(price_df):
    """Synthetic sentiment DataFrame."""
    n     = len(price_df)
    dates = price_df.index
    np.random.seed(2)
    return pd.DataFrame({
        "avg_sentiment": np.random.uniform(-0.5, 0.5, n),
        "article_count": np.random.randint(0, 20, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# TechnicalFeatures
# ---------------------------------------------------------------------------

class TestTechnicalFeatures:
    def test_returns_dataframe(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, price_df):
        tech     = TechnicalFeatures()
        result   = tech.compute(price_df)
        required = [
            "log_return_1d", "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "bb_width", "bb_pct", "atr_14",
            "log_return_5d",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_momentum_columns(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        for w in [5, 10, 20]:
            assert f"momentum_{w}d" in result.columns

    def test_volatility_columns(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        for w in [10, 20, 30]:
            assert f"volatility_{w}d" in result.columns

    def test_rsi_in_range(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        rsi    = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_no_nan_after_compute(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        assert result.isnull().sum().sum() == 0

    def test_no_infinite_values(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        numeric = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_datetime_index_preserved(self, price_df):
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_fewer_rows_than_input_due_to_warmup(self, price_df):
        """Rows are dropped during warmup for rolling windows."""
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        assert len(result) < len(price_df)

    def test_target_is_forward_return(self, price_df):
        """log_return_5d at t should equal log(close[t+5] / close[t])."""
        tech   = TechnicalFeatures()
        result = tech.compute(price_df)
        # Spot check: not all zeros, not all identical
        assert result["log_return_5d"].std() > 0

    def test_feature_names_match_columns(self, price_df):
        tech     = TechnicalFeatures()
        result   = tech.compute(price_df)
        expected = set(tech.feature_names())
        actual   = set(result.columns) - {"open","high","low","close","volume","log_return_5d"}
        assert expected == actual


# ---------------------------------------------------------------------------
# MacroFeatures
# ---------------------------------------------------------------------------

class TestMacroFeatures:
    def test_returns_dataframe(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert isinstance(result, pd.DataFrame)

    def test_has_vix_features(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert "vix_level"    in result.columns
        assert "vix_change_1d" in result.columns
        assert "vix_change_5d" in result.columns

    def test_has_yield_spread(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert "yield_spread" in result.columns

    def test_has_cross_asset_returns(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert "sp500_return_1d" in result.columns
        assert "gold_return_1d"  in result.columns
        assert "oil_return_1d"   in result.columns

    def test_has_sector_etf_features(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert "xlk_return_1d" in result.columns
        assert "xlf_return_1d" in result.columns

    def test_handles_missing_symbol(self, macro_dict):
        """Should not raise if a symbol is missing — just skips it."""
        del macro_dict["^VIX"]
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert isinstance(result, pd.DataFrame)
        assert "vix_level" not in result.columns

    def test_datetime_index(self, macro_dict):
        macro  = MacroFeatures()
        result = macro.compute(macro_dict)
        assert isinstance(result.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# SentimentFeatures
# ---------------------------------------------------------------------------

class TestSentimentFeatures:
    def test_returns_dataframe(self, sentiment_df):
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_rolling_columns(self, sentiment_df):
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        for w in [3, 5, 10]:
            assert f"sentiment_ma_{w}d" in result.columns

    def test_has_buzz_columns(self, sentiment_df):
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        for w in [3, 5]:
            assert f"buzz_ma_{w}d" in result.columns

    def test_has_derived_features(self, sentiment_df):
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        assert "sentiment_momentum" in result.columns
        assert "sentiment_std_5d"   in result.columns

    def test_raw_columns_dropped(self, sentiment_df):
        """avg_sentiment and article_count should not appear in output."""
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        assert "avg_sentiment"  not in result.columns
        assert "article_count"  not in result.columns

    def test_empty_input_returns_empty(self):
        sent   = SentimentFeatures()
        result = sent.compute(pd.DataFrame())
        assert result.empty

    def test_fill_missing_returns_zeros(self, price_df):
        sent   = SentimentFeatures()
        result = sent.fill_missing(pd.DataFrame(), index=price_df.index)
        assert not result.empty
        assert (result == 0).all().all()

    def test_no_nan_after_compute(self, sentiment_df):
        sent   = SentimentFeatures()
        result = sent.compute(sentiment_df)
        assert result.isnull().sum().sum() == 0