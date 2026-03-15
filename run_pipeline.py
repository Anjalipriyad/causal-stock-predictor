"""
run_pipeline.py
---------------
Single script to run the entire ML pipeline end to end.
Automatically detects what already exists and skips completed steps.

Usage:
    python run_pipeline.py --ticker AAPL              # smart run — skips what exists
    python run_pipeline.py --ticker AAPL --predict-only   # just live prediction
    python run_pipeline.py --ticker AAPL --refresh-data   # re-download data
    python run_pipeline.py --ticker AAPL --full-retrain   # retrain everything from scratch
    python run_pipeline.py --ticker AAPL --with-retrain   # add walk-forward retraining
    python run_pipeline.py --ticker AAPL --with-regime    # add regime-aware models
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# ── Make ml.src.* imports work ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def check_exists(path: Path) -> bool:
    """Return True if path exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


# ── Pipeline steps ───────────────────────────────────────────────────────────

def step1_load_data(ticker: str, skip: bool) -> None:
    banner(f"STEP 1 — Load Historical Data ({ticker})")
    if skip:
        logger.info("Data already on disk — skipping download.")
        return
    from ml.src.data.loader import DataLoader
    start  = time.time()
    loader = DataLoader()
    loader.load_historical(ticker)
    logger.info(f"Data loaded in {elapsed(start)}")


def step2_build_features(ticker: str, force: bool) -> None:
    banner(f"STEP 2 — Feature Engineering ({ticker})")
    from ml.src.features.pipeline import FeaturePipeline
    start    = time.time()
    pipeline = FeaturePipeline()
    df       = pipeline.build(ticker, force=force)
    logger.info(
        f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} cols "
        f"built in {elapsed(start)}"
    )


def step3_causal_discovery(ticker: str) -> list[str]:
    banner(f"STEP 3 — Causal Discovery ({ticker})")
    import pandas as pd
    from ml.src.data.loader import _load_config
    from ml.src.causal.granger import GrangerCausality
    from ml.src.causal.pcmci import PCMCIDiscovery
    from ml.src.causal.selector import CausalSelector
    from ml.src.features.pipeline import FeaturePipeline

    cfg       = _load_config()
    pipeline  = FeaturePipeline()
    feat_path = pipeline.features_dir / f"{ticker}_features.csv"
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    target    = cfg["model"]["target"]

    # Granger
    logger.info("Running Granger causality...")
    start            = time.time()
    granger          = GrangerCausality()
    granger_results  = granger.run(df, target=target, verbose=False)
    granger_features = granger.get_causal_features(granger_results)
    logger.info(f"Granger done in {elapsed(start)} — {len(granger_features)} causal features")

    # PCMCI — last 5 years only for speed
    logger.info("Running PCMCI on last 5 years (5-15 mins)...")
    start         = time.time()
    df_pcmci      = df.loc[str(pd.Timestamp.now().year - 5):]
    logger.info(f"PCMCI dataset: {len(df_pcmci)} rows")
    pcmci         = PCMCIDiscovery()
    pcmci_results = pcmci.run(df_pcmci, target=target)
    pcmci_features = pcmci.get_causal_features(pcmci_results)
    logger.info(f"PCMCI done in {elapsed(start)} — {len(pcmci_features)} causal features")

    # Select + save
    selector = CausalSelector()
    features = selector.select(ticker, granger_results, pcmci_results, save=True)

    print(f"\nFinal causal features ({len(features)}):")
    for i, f in enumerate(features, 1):
        print(f"  {i:2d}. {f}")
    return features


def step4_train_models(ticker: str, causal_features: list[str]) -> None:
    banner(f"STEP 4 — Train Models ({ticker})")
    import pandas as pd
    from ml.src.ensemble import Ensemble
    from ml.src.evaluation.metrics import Metrics
    from ml.src.features.pipeline import FeaturePipeline

    pipeline  = FeaturePipeline()
    feat_path = pipeline.features_dir / f"{ticker}_features.csv"
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    start          = time.time()
    ensemble       = Ensemble()
    X_test, y_test = ensemble.train_all(df, ticker, causal_features)
    logger.info(f"All models trained in {elapsed(start)}")

    test_df = pd.concat([X_test, y_test], axis=1)
    preds   = ensemble.predict_historical(test_df, causal_features)
    metrics = Metrics()
    scores  = metrics.compute_all(
        preds["predicted_return"],
        preds["actual_return"],
        label="Test set",
    )
    print(f"\n=== TEST SET RESULTS ===")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")


def step5_sample_prediction(ticker: str, causal_features: list[str]) -> None:
    banner(f"STEP 5 — Live Prediction & Model Report ({ticker})")
    import pandas as pd
    from ml.src.data.loader import DataLoader
    from ml.src.features.pipeline import FeaturePipeline
    from ml.src.ensemble import Ensemble
    from ml.src.evaluation.metrics import Metrics
    from ml.src.evaluation.regime_splitter import RegimeSplitter

    logger.info("Fetching live data...")
    loader   = DataLoader()
    live     = loader.load_live(ticker)

    logger.info("Building live feature vector...")
    pipeline = FeaturePipeline()
    row      = pipeline.build_live(ticker, live)

    logger.info("Running prediction...")
    ensemble = Ensemble()
    ensemble.load(ticker)

    current_price = float(live["prices"]["close"].iloc[-1])
    result = ensemble.predict_live(
        live_features=row,
        ticker=ticker,
        current_price=current_price,
        causal_features=causal_features,
    )

    # Compute model metrics from saved feature matrix
    model_metrics = {}
    feat_path = pipeline.features_dir / f"{ticker}_features.csv"
    if feat_path.exists():
        try:
            df    = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            preds = ensemble.predict_historical(df, causal_features)
            m     = Metrics()
            if "actual_return" in preds.columns:
                model_metrics["overall"] = m.compute_all(
                    preds["predicted_return"], preds["actual_return"], label="overall"
                )
                splitter = RegimeSplitter()
                for regime, rdf in splitter.split_all(df).items():
                    rp = preds.loc[preds.index.isin(rdf.index)]
                    if len(rp) >= 30 and "actual_return" in rp.columns:
                        model_metrics[regime] = m.compute_all(
                            rp["predicted_return"], rp["actual_return"], label=regime
                        )
        except Exception as e:
            logger.warning(f"Could not compute metrics: {e}")

    W = 60
    print(f"{'='*W}")
    print(f"  LIVE PREDICTION — {ticker}   ({result.prediction_date})")
    print(f"{'='*W}")
    print(f"  Current price:    ${result.current_price:.2f}")
    print(f"  Predicted price:  ${result.predicted_price:.2f}")
    print(f"  Direction:        {result.direction}")
    print(f"  Expected return:  {result.predicted_return:+.2%}")
    print(f"  Confidence:       {result.confidence:.0%}")
    print(f"  Range (90% CI):   ${result.lower_band:.2f} — ${result.upper_band:.2f}")
    print(f"  Horizon:          {result.horizon_days} trading days")
    print(f"  Model:            {result.model_name}")
    print(f"Causal Drivers:")
    for d in result.causal_drivers:
        shap  = d.get("shap", d["value"])
        arrow = "▲" if d["impact"] == "positive" else "▼"
        print(f"    {arrow} {d['feature']:30s} {d['impact']:8s}  ({shap:+.4f})")
    print(f"Causal Features Used ({len(causal_features)}):")
    print(f"    {', '.join(causal_features)}")
    if model_metrics and "overall" in model_metrics:
        regimes = [r for r in model_metrics if r != "overall"]
        print(f"Model Performance:")
        header = f"    {'Metric':<25} {'Overall':>10}" + "".join(f"  {r[:12]:>12}" for r in regimes)
        print(header)
        print("    " + "-"*25 + " " + "-"*10 + "".join("  " + "-"*12 for _ in regimes))
        for metric in ["directional_accuracy","sharpe_ratio","max_drawdown","calmar_ratio","rmse"]:
            row_str = f"    {metric:<25} {model_metrics['overall'].get(metric,0):>10.4f}"
            for regime in regimes:
                row_str += f"  {model_metrics[regime].get(metric,0):>12.4f}"
            print(row_str)
        print(f"Random baseline = 0.50  |  Target > 0.52")
    print(f"{'='*W}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="causal-stock-predictor — full ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First time:        python run_pipeline.py --ticker AAPL
  Daily prediction:  python run_pipeline.py --ticker AAPL --predict-only
  Re-download data:  python run_pipeline.py --ticker AAPL --refresh-data
  Full retrain:      python run_pipeline.py --ticker AAPL --full-retrain
  With retraining:   python run_pipeline.py --ticker AAPL --with-retrain
  With regime:       python run_pipeline.py --ticker AAPL --with-regime
        """
    )
    parser.add_argument("--ticker",        type=str, required=True,
                        help="Stock ticker, e.g. AAPL")
    parser.add_argument("--predict-only",  action="store_true",
                        help="Only run live prediction (models must exist)")
    parser.add_argument("--refresh-data",  action="store_true",
                        help="Re-download data even if already on disk")
    parser.add_argument("--full-retrain",  action="store_true",
                        help="Force rebuild features + causal discovery + retrain models")
    parser.add_argument("--with-retrain",  action="store_true",
                        help="Also run walk-forward retraining (~1hr)")
    parser.add_argument("--with-regime",   action="store_true",
                        help="Also train regime-aware models (~40min)")
    args   = parser.parse_args()
    ticker = args.ticker.upper()

    total_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  causal-stock-predictor")
    print(f"  Ticker: {ticker}")
    print(f"{'#'*60}")

    # ── Predict only ─────────────────────────────────────────────────────────
    if args.predict_only:
        from ml.src.causal.selector import CausalSelector
        causal_features = CausalSelector().load(ticker)
        step5_sample_prediction(ticker, causal_features)
        return

    # ── Auto-detect what already exists ──────────────────────────────────────
    from ml.src.features.pipeline import FeaturePipeline
    from ml.src.causal.selector import CausalSelector
    from ml.src.data.loader import _load_config

    cfg        = _load_config()
    pipeline   = FeaturePipeline()
    models_dir = ROOT / cfg["saved_models"]["dir"]
    feat_path  = pipeline.features_dir / f"{ticker}_features.csv"
    causal_path = models_dir / f"causal_features_{ticker}.json"
    lgbm_path  = models_dir / f"lgbm_{ticker}.pkl"
    data_path  = ROOT / cfg["data"]["raw_dir"] / "prices" / f"{ticker}.csv"

    # Decide what to skip
    skip_data   = check_exists(data_path)  and not args.refresh_data
    skip_feat   = check_exists(feat_path)  and not args.full_retrain
    skip_causal = check_exists(causal_path) and not args.full_retrain
    skip_train  = check_exists(lgbm_path)  and not args.full_retrain

    print(f"\n  Status:")
    print(f"    Raw data:        {'✓ exists' if skip_data   else '✗ missing'} {'(skipping)' if skip_data   else '(will download)'}")
    print(f"    Feature matrix:  {'✓ exists' if skip_feat   else '✗ missing'} {'(skipping)' if skip_feat   else '(will build)'}")
    print(f"    Causal features: {'✓ exists' if skip_causal else '✗ missing'} {'(skipping)' if skip_causal else '(will run PCMCI)'}")
    print(f"    Trained models:  {'✓ exists' if skip_train  else '✗ missing'} {'(skipping)' if skip_train  else '(will train)'}")

    try:
        # Step 1 — Data
        step1_load_data(ticker, skip=skip_data)

        # Step 2 — Features
        step2_build_features(ticker, force=not skip_feat)

        # Step 3 — Causal discovery
        if skip_causal:
            causal_features = CausalSelector().load(ticker)
            logger.info(f"Loaded saved causal features ({len(causal_features)}): {causal_features}")
        else:
            causal_features = step3_causal_discovery(ticker)

        # Step 4 — Train models
        if not skip_train:
            step4_train_models(ticker, causal_features)
        else:
            logger.info("Models already trained — skipping. Use --full-retrain to force.")

        # Step 5a — Walk-forward retraining (opt-in)
        if args.with_retrain:
            banner(f"STEP 5a — Walk-Forward Retraining ({ticker})")
            import pandas as pd
            from ml.src.evaluation.retrain_schedule import RetrainScheduler
            df_r        = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            scheduler   = RetrainScheduler()
            retrain_res = scheduler.run(df_r, ticker, causal_features, force=True)
            print(f"Retraining complete: {len(retrain_res)} windows evaluated.")

        # Step 5b — Regime-aware models (opt-in)
        if args.with_regime:
            banner(f"STEP 5b — Regime-Aware Training ({ticker})")
            import pandas as pd
            from ml.src.models.regime_model import RegimeAwareEnsemble
            df_rg        = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            regime_model = RegimeAwareEnsemble()
            regime_res   = regime_model.fit_all_regimes(df_rg, ticker, causal_features)
            print(f"Regime models trained: {list(regime_res.keys())}")

        # Step 6 — Live prediction
        step5_sample_prediction(ticker, causal_features)

        banner(f"PIPELINE COMPLETE — {elapsed(total_start)} total")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()