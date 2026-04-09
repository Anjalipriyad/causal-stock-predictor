"""
selection_ablation.py
---------------------
Compare causal selection strategies: intersection vs union vs adaptive.

Usage:
    python ml/scripts/selection_ablation.py --ticker AAPL
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ml.src.improvements.diagnostics import _load_feature_matrix
from ml.src.data.loader import _load_config
from ml.src.causal.granger import GrangerCausality
from ml.src.causal.pcmci import PCMCIDiscovery
from ml.src.causal.selector import CausalSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ablation")


def main():
    parser = argparse.ArgumentParser(description="Run causal selection ablation")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--market", choices=["us", "india"], default="us")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg = _load_config()

    # Load feature matrix
    df = _load_feature_matrix(ticker, market=args.market)

    # Choose target column (NIFTY / India use log_return_5d)
    if args.market == "india" or ticker in ("NIFTY", "^NSEI", "NIFTY50"):
        target = "log_return_5d"
        logger.info("Using log_return_5d as target (India/NIFTY)")
    else:
        target = cfg["model"]["target"]

    # Training slice (same logic as pipeline)
    n = len(df)
    train_end = int(n * cfg["model"]["train_ratio"]) if n > 0 else 0
    df_train = df.iloc[:train_end]
    if df_train.empty:
        raise RuntimeError("Not enough rows for training split — aborting ablation.")

    # Granger on full training set
    granger = GrangerCausality()
    logger.info("Running Granger causality on training set...")
    granger_results = granger.run(df_train, target=target, verbose=False)

    # PCMCI on last 50% of training set (consistent with pipeline)
    df_pcmci = df_train.iloc[-int(len(df_train) * 0.5):]
    pcmci = PCMCIDiscovery()
    try:
        logger.info("Running PCMCI on last 50% of training set...")
        pcmci_results = pcmci.run(df_pcmci, target=target)
    except Exception as e:
        logger.warning(f"PCMCI run failed: {e} — continuing with empty PCMCI results.")
        pcmci_results = {"causal_links": {}}

    selector = CausalSelector(cfg=cfg)

    results = {}

    # Intersection (strict)
    selector.strategy = "intersection"
    features_intersection = selector.select(ticker, granger_results, pcmci_results, save=False)
    info_intersection = getattr(selector, "_last_selection_info", None)
    results["intersection"] = {"n": len(features_intersection), "features": features_intersection, "info": info_intersection}

    # Union (permissive)
    selector.strategy = "union"
    features_union = selector.select(ticker, granger_results, pcmci_results, save=False)
    info_union = getattr(selector, "_last_selection_info", None)
    results["union"] = {"n": len(features_union), "features": features_union, "info": info_union}

    # Adaptive (force adaptive relaxations)
    selector.strategy = "intersection"
    features_adaptive = selector.select(ticker, granger_results, pcmci_results, save=False, force_adaptive=True)
    info_adaptive = getattr(selector, "_last_selection_info", None)
    results["adaptive"] = {"n": len(features_adaptive), "features": features_adaptive, "info": info_adaptive}

    # Print concise summary
    print(f"\nSelection ablation — {ticker} ({args.market})")
    for k, v in results.items():
        print(f"  {k:10s}: n={v['n']:2d}  features={v['features']}")

    # Save results to saved_models
    root = Path(__file__).resolve().parents[3]
    models_dir = root / cfg["saved_models"]["dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"selection_ablation_{ticker}.json"
    with open(out_path, "w") as f:
        json.dump({"ticker": ticker, "market": args.market, "results": results}, f, indent=2)

    print(f"\nSaved ablation results → {out_path}")


if __name__ == "__main__":
    main()
