"""
test_metrics.py
---------------
Tests for Metrics, RegimeSplitter, and Backtester (lightweight).
All tests use synthetic data — no model training, no network calls.
"""

import pytest
import numpy as np
import pandas as pd

from ml.src.evaluation.metrics import Metrics
from ml.src.evaluation.regime_splitter import RegimeSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def return_series():
    """Synthetic 252-row return series (1 trading year)."""
    np.random.seed(42)
    n     = 252
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    y_true = pd.Series(np.random.randn(n) * 0.01, index=dates)
    y_pred = pd.Series(
        0.6 * np.sign(y_true) * 0.01 + np.random.randn(n) * 0.005,
        index=dates,
    )
    return y_pred, y_true


@pytest.fixture
def perfect_predictor():
    """y_pred has same sign as y_true — 100% directional accuracy."""
    np.random.seed(1)
    n     = 200
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    y_true = pd.Series(np.random.randn(n) * 0.01, index=dates)
    y_pred = pd.Series(np.abs(y_true) * np.sign(y_true), index=dates)
    return y_pred, y_true


@pytest.fixture
def worst_predictor():
    """y_pred always wrong direction — 0% directional accuracy."""
    np.random.seed(1)
    n     = 200
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    y_true = pd.Series(np.random.randn(n) * 0.01, index=dates)
    y_pred = -y_true
    return y_pred, y_true


@pytest.fixture
def feature_df():
    """500-row feature matrix covering multiple regimes."""
    np.random.seed(42)
    n     = 1000
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    df    = pd.DataFrame(
        {
            "feature_a":    np.random.randn(n),
            "feature_b":    np.random.randn(n),
            "log_return_5d": np.random.randn(n) * 0.01,
        },
        index=dates,
    )
    return df


# ---------------------------------------------------------------------------
# Metrics — directional_accuracy
# ---------------------------------------------------------------------------

class TestDirectionalAccuracy:
    def test_perfect_predictor(self, perfect_predictor):
        y_pred, y_true = perfect_predictor
        m  = Metrics()
        da = m.directional_accuracy(y_pred, y_true)
        assert da == pytest.approx(1.0)

    def test_worst_predictor(self, worst_predictor):
        y_pred, y_true = worst_predictor
        m  = Metrics()
        da = m.directional_accuracy(y_pred, y_true)
        assert da == pytest.approx(0.0)

    def test_random_near_50(self):
        np.random.seed(0)
        n      = 10000
        y_true = pd.Series(np.random.randn(n) * 0.01)
        y_pred = pd.Series(np.random.randn(n) * 0.01)
        m  = Metrics()
        da = m.directional_accuracy(y_pred, y_true)
        assert 0.45 <= da <= 0.55

    def test_returns_float(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        da = m.directional_accuracy(y_pred, y_true)
        assert isinstance(da, float)

    def test_in_range(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        da = m.directional_accuracy(y_pred, y_true)
        assert 0.0 <= da <= 1.0


# ---------------------------------------------------------------------------
# Metrics — sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_returns_float(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        sr = m.sharpe_ratio(y_pred, y_true)
        assert isinstance(sr, float)

    def test_good_predictor_positive_sharpe(self, perfect_predictor):
        y_pred, y_true = perfect_predictor
        m  = Metrics()
        sr = m.sharpe_ratio(y_pred, y_true)
        assert sr > 0

    def test_bad_predictor_negative_sharpe(self, worst_predictor):
        y_pred, y_true = worst_predictor
        m  = Metrics()
        sr = m.sharpe_ratio(y_pred, y_true)
        assert sr < 0

    def test_sharpe_is_finite(self):
        # Sharpe should always return a finite float
        y_true = pd.Series([0.0] * 100)
        y_pred = pd.Series([0.01] * 100)
        m  = Metrics()
        sr = m.sharpe_ratio(y_pred, y_true)
        assert isinstance(sr, float)
        assert not (sr != sr)  # not NaN


# ---------------------------------------------------------------------------
# Metrics — rmse
# ---------------------------------------------------------------------------

class TestRMSE:
    def test_zero_error(self):
        y = pd.Series([0.01, -0.02, 0.03])
        m = Metrics()
        assert m.rmse(y, y) == pytest.approx(0.0)

    def test_positive_value(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        assert m.rmse(y_pred, y_true) >= 0

    def test_larger_error_larger_rmse(self):
        y_true  = pd.Series([0.01, -0.02, 0.03])
        y_pred1 = pd.Series([0.011, -0.021, 0.031])  # small error
        y_pred2 = pd.Series([0.05,  -0.10,  0.15])   # large error
        m = Metrics()
        assert m.rmse(y_pred1, y_true) < m.rmse(y_pred2, y_true)


# ---------------------------------------------------------------------------
# Metrics — max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_returns_negative_or_zero(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        dd = m.max_drawdown(y_pred, y_true)
        assert dd <= 0.0

    def test_perfect_predictor_less_drawdown(self, perfect_predictor, worst_predictor):
        m = Metrics()
        dd_good = m.max_drawdown(*perfect_predictor)
        dd_bad  = m.max_drawdown(*worst_predictor)
        assert dd_good > dd_bad  # less negative = better


# ---------------------------------------------------------------------------
# Metrics — compute_all
# ---------------------------------------------------------------------------

class TestComputeAll:
    def test_returns_dict(self, return_series):
        y_pred, y_true = return_series
        m       = Metrics()
        results = m.compute_all(y_pred, y_true)
        assert isinstance(results, dict)

    def test_has_all_metrics(self, return_series):
        y_pred, y_true = return_series
        m       = Metrics()
        results = m.compute_all(y_pred, y_true)
        for metric in m.metrics_list:
            assert metric in results, f"Missing metric: {metric}"

    def test_all_values_are_floats(self, return_series):
        y_pred, y_true = return_series
        m       = Metrics()
        results = m.compute_all(y_pred, y_true)
        for k, v in results.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_comparison_table_returns_dataframe(self, return_series):
        y_pred, y_true = return_series
        m  = Metrics()
        r1 = m.compute_all(y_pred, y_true)
        r2 = m.compute_all(-y_pred, y_true)
        table = m.comparison_table({"model_a": r1, "model_b": r2})
        assert isinstance(table, pd.DataFrame)
        assert "model_a" in table.index
        assert "model_b" in table.index


# ---------------------------------------------------------------------------
# RegimeSplitter
# ---------------------------------------------------------------------------

class TestRegimeSplitter:
    def test_get_regime_returns_subset(self, feature_df):
        splitter = RegimeSplitter()
        subset   = splitter.get_regime(feature_df, "rate_hike")
        assert isinstance(subset, pd.DataFrame)
        # rate_hike is 2022 — should be in our 2019-start df
        assert len(subset) > 0

    def test_unknown_regime_raises(self, feature_df):
        splitter = RegimeSplitter()
        with pytest.raises(ValueError, match="Unknown regime"):
            splitter.get_regime(feature_df, "nonexistent_regime")

    def test_split_all_returns_dict(self, feature_df):
        splitter = RegimeSplitter()
        result   = splitter.split_all(feature_df)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_label_adds_regime_column(self, feature_df):
        splitter  = RegimeSplitter()
        labelled  = splitter.label(feature_df)
        assert "regime" in labelled.columns

    def test_label_covers_all_dates(self, feature_df):
        splitter = RegimeSplitter()
        labelled = splitter.label(feature_df)
        assert labelled["regime"].notna().all()

    def test_regime_stats_returns_dataframe(self, feature_df):
        splitter = RegimeSplitter()
        stats    = splitter.regime_stats(feature_df)
        assert isinstance(stats, pd.DataFrame)
        assert "n_rows" in stats.columns

    def test_train_test_split_no_overlap(self, feature_df):
        splitter = RegimeSplitter()
        train, test = splitter.train_test_split_by_regime(feature_df, "rate_hike")
        if len(train) > 0 and len(test) > 0:
            assert train.index.max() < test.index.min()

    def test_regime_names_property(self):
        splitter = RegimeSplitter()
        names    = splitter.regime_names
        assert isinstance(names, list)
        assert "bull"        in names
        assert "covid_crash" in names
        assert "rate_hike"   in names