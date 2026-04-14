"""
test_models.py
--------------
Tests for BaseModel/PredictionResult, LGBMModel, XGBModel, ARIMAModel, Ensemble.
All tests use synthetic data — no disk I/O except tmp_path fixtures.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from ml.src.models.base_model import BaseModel, PredictionResult
from ml.src.models.lgbm_model import LGBMModel
from ml.src.models.xgb_model import XGBModel
from ml.src.models.arima_model import ARIMAModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """
    500 rows, 5 causal features, 1 target.
    Enough for train/val/test split + early stopping.
    """
    np.random.seed(42)
    n     = 500
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    X     = pd.DataFrame(
        np.random.randn(n, 5),
        columns=["f1", "f2", "f3", "volatility_10d", "f5"],
        index=pd.DatetimeIndex(dates, name="date"),
    )
    # Make f1 + f2 mildly predictive
    y = pd.Series(
        0.3 * X["f1"] + 0.2 * X["f2"] + np.random.randn(n) * 0.5,
        index=X.index, name="log_return_5d",
    )
    return X, y


@pytest.fixture
def split_data(synthetic_data):
    """Pre-split train/val/test."""
    X, y  = synthetic_data
    n     = len(X)
    t1    = int(n * 0.7)
    t2    = int(n * 0.85)
    return (
        X.iloc[:t1],  X.iloc[t1:t2], X.iloc[t2:],
        y.iloc[:t1],  y.iloc[t1:t2], y.iloc[t2:],
    )


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    @pytest.fixture
    def sample_result(self):
        return PredictionResult(
            ticker="AAPL",
            predicted_return=0.023,
            predicted_price=218.0,
            current_price=213.0,
            direction="UP",
            confidence=0.65,
            upper_band=222.0,
            lower_band=214.0,
            causal_drivers=[{"feature": "vix", "impact": "negative", "value": -0.5}],
            model_name="lgbm",
            horizon_days=5,
            prediction_date="2024-01-15",
        )

    def test_to_dict_returns_dict(self, sample_result):
        d = sample_result.to_dict()
        assert isinstance(d, dict)
        assert d["ticker"] == "AAPL"

    def test_to_json_is_valid_json(self, sample_result):
        import json
        js = sample_result.to_json()
        parsed = json.loads(js)
        assert parsed["ticker"] == "AAPL"

    def test_from_dict_roundtrip(self, sample_result):
        d       = sample_result.to_dict()
        result2 = PredictionResult.from_dict(d)
        assert result2.ticker           == sample_result.ticker
        assert result2.predicted_return == sample_result.predicted_return
        assert result2.direction        == sample_result.direction

    def test_str_representation(self, sample_result):
        s = str(sample_result)
        assert "AAPL" in s
        assert "UP"   in s

    def test_direction_is_up_or_down(self, sample_result):
        assert sample_result.direction in {"UP", "DOWN"}

    def test_confidence_in_range(self, sample_result):
        assert 0.0 <= sample_result.confidence <= 1.0


# ---------------------------------------------------------------------------
# LGBMModel
# ---------------------------------------------------------------------------

class TestLGBMModel:
    def test_fit_sets_is_fitted(self, split_data):
        X_tr, X_va, _, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        assert model._is_fitted is True

    def test_predict_raw_returns_array(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        preds = model.predict_raw(X_te)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_te)

    def test_predict_returns_prediction_result(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        result = model.predict(X_te, ticker="AAPL", current_price=200.0)
        assert isinstance(result, PredictionResult)

    def test_direction_is_valid(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        result = model.predict(X_te, ticker="AAPL", current_price=200.0)
        assert result.direction in {"UP", "DOWN"}

    def test_save_and_load(self, tmp_path, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.models_dir = tmp_path
        model.fit(X_tr, y_tr, X_va, y_va)
        model.save("AAPL")

        model2 = LGBMModel()
        model2.models_dir = tmp_path
        model2.load("AAPL")
        assert model2._is_fitted is True

        preds1 = model.predict_raw(X_te)
        preds2 = model2.predict_raw(X_te)
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_predict_before_fit_raises(self, split_data):
        _, _, X_te, _, _, _ = split_data
        model = LGBMModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_raw(X_te)

    def test_feature_importance_returns_series(self, split_data):
        X_tr, X_va, _, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X_tr.shape[1]


# ---------------------------------------------------------------------------
# XGBModel
# ---------------------------------------------------------------------------

class TestXGBModel:
    def test_fit_sets_is_fitted(self, split_data):
        X_tr, X_va, _, y_tr, y_va, _ = split_data
        model = XGBModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        assert model._is_fitted is True

    def test_predict_raw_returns_array(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = XGBModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        preds = model.predict_raw(X_te)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_te)

    def test_save_and_load(self, tmp_path, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = XGBModel()
        model.models_dir = tmp_path
        model.fit(X_tr, y_tr, X_va, y_va)
        model.save("AAPL")

        model2 = XGBModel()
        model2.models_dir = tmp_path
        model2.load("AAPL")
        assert model2._is_fitted is True

    def test_load_nonexistent_raises(self, tmp_path):
        model = XGBModel()
        model.models_dir = tmp_path
        with pytest.raises(FileNotFoundError):
            model.load("FAKE")


# ---------------------------------------------------------------------------
# ARIMAModel
# ---------------------------------------------------------------------------

class TestARIMAModel:
    def test_fit_sets_is_fitted(self, split_data):
        X_tr, _, _, y_tr, _, _ = split_data
        model = ARIMAModel()
        model.fit(X_tr, y_tr)
        assert model._is_fitted is True

    def test_predict_raw_returns_array(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = ARIMAModel()
        model.fit(X_tr, y_tr)
        preds = model.predict_raw(X_te)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X_te)

    def test_all_predictions_same_value(self, split_data):
        """ARIMA should produce rolling forecasts (not a single repeated value)."""
        X_tr, _, X_te, y_tr, _, y_te = split_data
        model = ARIMAModel()
        model.fit(X_tr, y_tr)
        # Provide the actual held-out test targets so predict_val_set can produce
        # genuine one-step-ahead rolling forecasts for the validation/test period.
        preds = model.predict_val_set(y_te)
        # Basic checks: length matches and values are finite
        assert len(preds) == len(y_te)
        assert np.all(np.isfinite(preds))
        # If the fitted ARIMA is non-trivial (not constant mean ARIMA(0,0,0)),
        # expect non-constant rolling forecasts. If ARIMA(0,0,0) was selected
        # by auto-arima, constant predictions are a valid outcome.
        model_order = getattr(getattr(model, "_model", None), "order", None)
        if model_order is None or tuple(model_order) != (0, 0, 0):
            assert not np.allclose(preds, preds[0])

    def test_save_and_load(self, tmp_path, split_data):
        X_tr, _, _, y_tr, _, _ = split_data
        model = ARIMAModel()
        model.models_dir = tmp_path
        model.fit(X_tr, y_tr)
        model.save("AAPL")

        model2 = ARIMAModel()
        model2.models_dir = tmp_path
        model2.load("AAPL")
        assert model2._is_fitted is True

    def test_model_summary_returns_string(self, split_data):
        X_tr, _, _, y_tr, _, _ = split_data
        model = ARIMAModel()
        model.fit(X_tr, y_tr)
        s = model.model_summary()
        assert isinstance(s, str)
        assert len(s) > 0


# ---------------------------------------------------------------------------
# BaseModel helpers
# ---------------------------------------------------------------------------

class TestBaseModelHelpers:
    def test_prepare_data_shapes(self, synthetic_data):
        X, y  = synthetic_data
        df    = X.copy()
        df["log_return_5d"] = y
        model = LGBMModel()
        X_tr, X_va, X_te, y_tr, y_va, y_te = model.prepare_data(
            df, causal_features=list(X.columns)
        )
        n = len(df)
        assert len(X_tr) == int(n * 0.70)
        assert len(X_va) == int(n * 0.85) - int(n * 0.70)
        assert len(X_te) == n - int(n * 0.85)

    def test_prepare_data_no_overlap(self, synthetic_data):
        X, y  = synthetic_data
        df    = X.copy()
        df["log_return_5d"] = y
        model = LGBMModel()
        X_tr, X_va, X_te, *_ = model.prepare_data(
            df, causal_features=list(X.columns)
        )
        assert X_tr.index.max() < X_va.index.min()
        assert X_va.index.max() < X_te.index.min()

    def test_confidence_in_range(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        result = model.predict(X_te, ticker="AAPL", current_price=200.0)
        assert 0.3 <= result.confidence <= 0.9

    def test_upper_band_above_lower(self, split_data):
        X_tr, X_va, X_te, y_tr, y_va, _ = split_data
        model = LGBMModel()
        model.fit(X_tr, y_tr, X_va, y_va)
        result = model.predict(X_te, ticker="AAPL", current_price=200.0)
        assert result.upper_band > result.lower_band