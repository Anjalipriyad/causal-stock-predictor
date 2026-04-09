"""
meta_tune.py
-------------
Train an alternative meta-learner that optimizes directional accuracy
by treating the meta problem as a classification task.

This script trains a simple logistic regression on validation-set
predictions from the base learners (lgbm, xgb, arima) and saves the
classifier to `saved_models/meta_classifier_{TICKER}.pkl`.
"""

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.src.data.loader import _load_config
from ml.src.ensemble import Ensemble
from ml.src.causal.selector import CausalSelector
from ml.src.improvements.diagnostics import _load_feature_matrix

logger = logging.getLogger(__name__)


def train_meta_classifier(ticker: str, market: str = "us") -> Dict:
    ticker = ticker.upper()
    cfg = _load_config()

    df = _load_feature_matrix(ticker, market)
    try:
        causal_features = CausalSelector().load(ticker)
    except Exception:
        causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != cfg["model"]["target"]][:15]

    ensemble = Ensemble(cfg=cfg)
    # Ensure base models exist — train if necessary
    try:
        ensemble.load(ticker)
    except Exception:
        logger.info("Training base models to obtain validation predictions.")
        ensemble.train_all(df, ticker, causal_features)

    # Prepare data splits using LGBM helper (same as Ensemble.train_all)
    X_train, X_val, X_test, y_train, y_val, y_test = ensemble.lgbm.prepare_data(df, causal_features)
    X_train_s, X_val_s, X_test_s = ensemble.lgbm.scale(X_train, X_val, X_test, ticker)

    # Validation-set predictions from base learners
    val_lgbm = ensemble.lgbm.predict_raw(X_val_s)
    val_xgb  = ensemble.xgb.predict_raw(X_val_s)
    # Give ARIMA the true validation targets so it can produce rolling forecasts
    val_arima = ensemble.arima.predict_raw(X_val, y_true=y_val)

    X_meta = np.column_stack([val_lgbm, val_xgb, val_arima])
    scaler = StandardScaler().fit(X_meta)
    X_meta_s = scaler.transform(X_meta)

    y_dir = (y_val.values >= 0).astype(int)

    clf = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=cfg["project"]["random_seed"]) 
    clf.fit(X_meta_s, y_dir)

    # Save classifier + scaler
    models_dir = Path(ensemble.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"meta_classifier_{ticker}.pkl"
    joblib.dump({"clf": clf, "scaler": scaler}, out_path)

    acc = float((clf.predict(X_meta_s) == y_dir).mean())
    logger.info(f"Saved meta-classifier → {out_path} (train-val accuracy={acc:.4f})")
    return {"ticker": ticker, "accuracy": acc, "path": str(out_path)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train classification meta-learner for directional accuracy")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    args = parser.parse_args()
    print(train_meta_classifier(args.ticker, market=args.market))
