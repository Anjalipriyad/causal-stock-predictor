"""
hpo.py
-------
Wrapper to run hyperparameter tuning (Optuna) using the existing
HyperparameterTuner integration inside the Ensemble training loop.

This helper sets an in-memory config override enabling Optuna and
automatically invokes ensemble.train_all() which triggers tuning.
"""

import logging
from copy import deepcopy
from typing import Dict

from ml.src.data.loader import _load_config
from ml.src.ensemble import Ensemble
from ml.src.causal.selector import CausalSelector
from ml.src.improvements.diagnostics import _load_feature_matrix

logger = logging.getLogger(__name__)


def run_hpo(ticker: str, market: str = "us", n_trials: int = 50) -> Dict:
    """
    Run Optuna hyperparameter tuning by invoking Ensemble.train_all with
    an in-memory config override that enables optuna.

    Results (best params) are saved to saved_models/tuning/ by the tuner.
    """
    ticker = ticker.upper()
    cfg = _load_config()
    cfg2 = deepcopy(cfg)
    cfg2["features"]["use_optuna"] = True
    cfg2["features"]["optuna_trials"] = n_trials

    df = _load_feature_matrix(ticker, market)
    try:
        causal_features = CausalSelector().load(ticker)
    except Exception:
        # fallback to numeric features
        causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != cfg2["model"]["target"]][:15]

    ensemble = Ensemble(cfg=cfg2)
    # Train_all will invoke the HyperparameterTuner when enabled
    ensemble.train_all(df, ticker, causal_features)

    logger.info(f"HPO run complete for {ticker}. Check saved_models/tuning for results.")
    return {"status": "ok", "ticker": ticker, "trials": n_trials}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HPO for a ticker")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    print(run_hpo(args.ticker, market=args.market, n_trials=args.trials))
