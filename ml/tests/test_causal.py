"""
test_causal.py
--------------
Tests for GrangerCausality, PCMCIDiscovery, CausalSelector.
Uses small synthetic datasets — fast, no network calls.
"""

import json
import pytest
import numpy as np
import pandas as pd

from ml.src.causal.granger import GrangerCausality
from ml.src.causal.pcmci import PCMCIDiscovery
from ml.src.causal.selector import CausalSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def causal_df():
    """
    Synthetic dataset with known causal structure:
        x1 → target (lagged by 1)
        x2 → no causal link (pure noise)
    300 rows — enough for stable Granger tests.
    """
    np.random.seed(42)
    n      = 300
    dates  = pd.date_range("2018-01-01", periods=n, freq="B")
    x1     = np.random.randn(n)
    x2     = np.random.randn(n)                  # pure noise
    target = np.roll(x1, 1) * 0.5 + np.random.randn(n) * 0.3

    return pd.DataFrame(
        {"log_return_5d": target, "x1": x1, "x2": x2},
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def multi_feature_df():
    """Larger synthetic dataset with multiple causal features."""
    np.random.seed(7)
    n     = 500
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    x1    = np.random.randn(n)
    x2    = np.random.randn(n)
    x3    = np.random.randn(n)
    noise = np.random.randn(n) * 0.2

    # x1 and x3 are causal, x2 is noise
    target = 0.4 * np.roll(x1, 1) + 0.3 * np.roll(x3, 2) + noise

    return pd.DataFrame(
        {
            "log_return_5d": target,
            "x1": x1, "x2": x2, "x3": x3,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


# ---------------------------------------------------------------------------
# GrangerCausality
# ---------------------------------------------------------------------------

class TestGrangerCausality:
    def test_run_returns_dict(self, causal_df):
        granger = GrangerCausality()
        results = granger.run(causal_df)
        assert isinstance(results, dict)

    def test_keys_are_feature_names(self, causal_df):
        granger  = GrangerCausality()
        results  = granger.run(causal_df)
        expected = {"x1", "x2"}
        assert set(results.keys()) == expected

    def test_result_has_required_fields(self, causal_df):
        granger = GrangerCausality()
        results = granger.run(causal_df)
        for name, r in results.items():
            assert "causal"   in r
            assert "min_pval" in r
            assert "best_lag" in r

    def test_pval_in_range(self, causal_df):
        granger = GrangerCausality()
        results = granger.run(causal_df)
        for r in results.values():
            assert 0.0 <= r["min_pval"] <= 1.0

    def test_causal_field_is_bool(self, causal_df):
        granger = GrangerCausality()
        results = granger.run(causal_df)
        for r in results.values():
            assert isinstance(r["causal"], bool)

    def test_x1_detected_as_causal(self, causal_df):
        """x1 is built to Granger-cause target — should be detected."""
        granger = GrangerCausality()
        results = granger.run(causal_df)
        assert results["x1"]["causal"] is True

    def test_missing_target_raises(self, causal_df):
        granger = GrangerCausality()
        with pytest.raises(ValueError, match="not found"):
            granger.run(causal_df, target="nonexistent_col")

    def test_get_causal_features_returns_list(self, causal_df):
        granger  = GrangerCausality()
        results  = granger.run(causal_df)
        features = granger.get_causal_features(results)
        assert isinstance(features, list)

    def test_get_causal_features_sorted_by_pval(self, multi_feature_df):
        granger  = GrangerCausality()
        results  = granger.run(multi_feature_df)
        features = granger.get_causal_features(results)
        # Check sorted ascending by p-value
        pvals = [results[f]["min_pval"] for f in features]
        assert pvals == sorted(pvals)

    def test_summary_table_returns_dataframe(self, causal_df):
        granger = GrangerCausality()
        results = granger.run(causal_df)
        table   = granger.summary_table(results)
        assert isinstance(table, pd.DataFrame)
        assert "feature" in table.columns
        assert "causal"  in table.columns


# ---------------------------------------------------------------------------
# PCMCIDiscovery
# ---------------------------------------------------------------------------

class TestPCMCIDiscovery:
    def test_run_returns_dict(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        assert isinstance(results, dict)

    def test_result_has_required_keys(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        assert "causal_links" in results
        assert "p_matrix"     in results
        assert "val_matrix"   in results
        assert "var_names"    in results
        assert "target"       in results

    def test_causal_links_excludes_target(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        assert "log_return_5d" not in results["causal_links"]

    def test_causal_links_have_required_fields(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        for name, info in results["causal_links"].items():
            assert "causal"   in info
            assert "pval"     in info
            assert "val"      in info
            assert "best_lag" in info

    def test_pvals_in_range(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        for info in results["causal_links"].values():
            assert 0.0 <= info["pval"] <= 1.0

    def test_missing_target_raises(self, causal_df):
        pcmci = PCMCIDiscovery()
        with pytest.raises(ValueError, match="not in DataFrame"):
            pcmci.run(causal_df, target="nonexistent")

    def test_get_causal_features_returns_list(self, causal_df):
        pcmci    = PCMCIDiscovery()
        results  = pcmci.run(causal_df)
        features = pcmci.get_causal_features(results)
        assert isinstance(features, list)

    def test_summary_table_returns_dataframe(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        table   = pcmci.summary_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_causal_graph_matrix_shape(self, causal_df):
        pcmci   = PCMCIDiscovery()
        results = pcmci.run(causal_df)
        matrix  = pcmci.causal_graph_matrix(results)
        n_vars  = len(results["var_names"])
        assert matrix.shape == (n_vars, n_vars)


# ---------------------------------------------------------------------------
# CausalSelector
# ---------------------------------------------------------------------------

class TestCausalSelector:
    @pytest.fixture
    def granger_results(self):
        return {
            "x1": {"causal": True,  "min_pval": 0.01, "best_lag": 1},
            "x2": {"causal": False, "min_pval": 0.45, "best_lag": 2},
            "x3": {"causal": True,  "min_pval": 0.03, "best_lag": 1},
        }

    @pytest.fixture
    def pcmci_results(self):
        return {
            "causal_links": {
                "x1": {"causal": True,  "pval": 0.008, "val": 0.32, "best_lag": 1},
                "x2": {"causal": False, "pval": 0.52,  "val": 0.01, "best_lag": 3},
                "x3": {"causal": True,  "pval": 0.04,  "val": 0.21, "best_lag": 2},
            },
            "p_matrix":   np.ones((4, 4, 6)),
            "val_matrix": np.zeros((4, 4, 6)),
            "var_names":  ["log_return_5d", "x1", "x2", "x3"],
            "target":     "log_return_5d",
        }

    def test_intersection_returns_both_causal(
        self, tmp_path, granger_results, pcmci_results
    ):
        selector = CausalSelector()
        selector.models_dir = tmp_path
        selector.min_causal_features = 2
        features = selector.select(
            "AAPL", granger_results, pcmci_results, save=False
        )
        # x1 and x3 are causal in both — intersection
        assert "x1" in features
        assert "x3" in features
        assert "x2" not in features

    def test_union_returns_any_causal(
        self, tmp_path, granger_results, pcmci_results
    ):
        selector = CausalSelector()
        selector.models_dir  = tmp_path
        selector.strategy    = "union"
        selector.min_causal_features = 2
        features = selector.select(
            "AAPL", granger_results, pcmci_results, save=False
        )
        # x1, x3 causal in both; x2 in neither
        assert "x1" in features
        assert "x3" in features

    def test_saves_json(self, tmp_path, granger_results, pcmci_results):
        selector = CausalSelector()
        selector.models_dir = tmp_path
        selector.min_causal_features = 2
        selector.select("AAPL", granger_results, pcmci_results, save=True)
        path = tmp_path / "causal_features_AAPL.json"
        assert path.exists()

    def test_json_structure(self, tmp_path, granger_results, pcmci_results):
        selector = CausalSelector()
        selector.models_dir = tmp_path
        selector.min_causal_features = 2
        selector.select("AAPL", granger_results, pcmci_results, save=True)
        with open(tmp_path / "causal_features_AAPL.json") as f:
            record = json.load(f)
        assert "ticker"     in record
        assert "strategy"   in record
        assert "n_features" in record
        assert "features"   in record
        assert record["ticker"] == "AAPL"

    def test_load_returns_feature_names(self, tmp_path, granger_results, pcmci_results):
        selector = CausalSelector()
        selector.models_dir = tmp_path
        selector.min_causal_features = 2
        selector.select("AAPL", granger_results, pcmci_results, save=True)
        loaded = selector.load("AAPL")
        assert isinstance(loaded, list)
        assert len(loaded) > 0

    def test_load_raises_if_not_saved(self, tmp_path):
        selector = CausalSelector()
        selector.models_dir = tmp_path
        with pytest.raises(FileNotFoundError):
            selector.load("MSFT")

    def test_min_features_raises(self, tmp_path, pcmci_results):
        """Should raise if fewer than min_causal_features survive."""
        granger_all_false = {
            k: {"causal": False, "min_pval": 0.9, "best_lag": 1}
            for k in ["x1", "x2", "x3"]
        }
        selector = CausalSelector()
        selector.models_dir = tmp_path
        with pytest.raises(ValueError, match="causal features found"):
            selector.select("AAPL", granger_all_false, pcmci_results, save=False)

    def test_comparison_table_returns_dataframe(
        self, granger_results, pcmci_results
    ):
        selector = CausalSelector()
        table    = selector.comparison_table(granger_results, pcmci_results)
        assert isinstance(table, pd.DataFrame)
        assert "feature"        in table.columns
        assert "granger_causal" in table.columns
        assert "pcmci_causal"   in table.columns