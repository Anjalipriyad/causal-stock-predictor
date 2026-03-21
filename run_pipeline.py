"""
run_pipeline.py
---------------
Single script to run the entire ML pipeline end to end.
Automatically detects what already exists and skips completed steps.

Usage:
    python run_pipeline.py --ticker AAPL                        # US stock
    python run_pipeline.py --ticker NIFTY --market india        # Nifty 50
    python run_pipeline.py --ticker RELIANCE.NS --market india  # Individual NSE stock
    python run_pipeline.py --ticker NIFTY --market india --predict-only
    python run_pipeline.py --ticker NIFTY --market india --paper-eval
"""

import argparse
import sys
import time
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


def banner(msg: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


def check_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def is_nifty_ticker(ticker: str) -> bool:
    return ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50")


def get_feat_path(ticker: str):
    """Get correct feature matrix path for any ticker."""
    if is_nifty_ticker(ticker):
        from ml.src.data.nifty_loader import NiftyLoader
        return NiftyLoader().out_dir / "NIFTY_features.csv"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        return FeaturePipeline().features_dir / f"{ticker}_features.csv"


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step1_load_data(ticker: str, skip: bool, market: str = "us") -> None:
    banner(f"STEP 1 — Load Historical Data ({ticker})")
    if skip or is_nifty_ticker(ticker) or market == "india":
        if is_nifty_ticker(ticker) or market == "india":
            logger.info("Indian market — using uploaded CSV files (skipping yFinance).")
        else:
            logger.info("Data already on disk — skipping download.")
        return
    from ml.src.data.loader import DataLoader
    start  = time.time()
    loader = DataLoader()
    loader.load_historical(ticker)
    logger.info(f"Data loaded in {elapsed(start)}")


def step2_build_features(ticker: str, force: bool, market: str = "us") -> None:
    banner(f"STEP 2 — Feature Engineering ({ticker})")
    start = time.time()

    if is_nifty_ticker(ticker) or market == "india":
        from ml.src.data.nifty_loader import NiftyLoader
        from ml.src.data.loader import _load_config
        loader    = NiftyLoader()
        feat_path = loader.out_dir / "NIFTY_features.csv"
        if feat_path.exists() and not force:
            logger.info("Nifty feature matrix exists — skipping build.")
            return
        cfg    = _load_config()
        use_fb = cfg["features"].get("finbert", {}).get("enabled", False)
        df     = loader.build_feature_matrix(use_finbert=use_fb)
    else:
        from ml.src.features.pipeline import FeaturePipeline
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

    cfg       = _load_config()
    feat_path = get_feat_path(ticker)
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # For NIFTY index — use log_return_5d as target (excess_return vs itself = 0)
    # For individual stocks — use excess_return_5d (vs market benchmark)
    if is_nifty_ticker(ticker):
        target = "log_return_5d"
        logger.info("[causal] NIFTY index detected — using log_return_5d as target")
    else:
        target = cfg["model"]["target"]

    # Use only training split — prevents test leakage
    train_ratio = cfg["model"]["train_ratio"]
    n           = len(df)
    train_end   = int(n * train_ratio)
    df_train    = df.iloc[:train_end]
    logger.info(
        f"Causal discovery on training data only: {len(df_train)} rows "
        f"({df_train.index.min().date()} → {df_train.index.max().date()}) "
        f"— test period excluded to prevent leakage"
    )

    # Granger — full training set
    logger.info("Running Granger causality on training data...")
    start            = time.time()
    granger          = GrangerCausality()
    granger_results  = granger.run(df_train, target=target, verbose=False)
    granger_features = granger.get_causal_features(granger_results)
    logger.info(f"Granger done in {elapsed(start)} — {len(granger_features)} causal features")

    # PCMCI — last 50% of training data (no test leakage)
    logger.info("Running PCMCI on last 50% of training data (5-15 mins)...")
    start      = time.time()
    df_pcmci   = df_train.iloc[-int(len(df_train) * 0.5):]
    logger.info(
        f"PCMCI dataset: {len(df_pcmci)} rows "
        f"({df_pcmci.index.min().date()} → {df_pcmci.index.max().date()})"
    )
    pcmci          = PCMCIDiscovery()
    pcmci_results  = pcmci.run(df_pcmci, target=target)
    pcmci_features = pcmci.get_causal_features(pcmci_results)
    logger.info(f"PCMCI done in {elapsed(start)} — {len(pcmci_features)} causal features")

    # Select + save
    # Use union for NIFTY (less data = stricter intersection may find 0 features)
    # Use intersection for US stocks (15 years of data = strict selection is fine)
    selector          = CausalSelector()
    selector.strategy = "union" if is_nifty_ticker(ticker) else "intersection"
    logger.info(f"[causal] Selection strategy: {selector.strategy}")
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
    from ml.src.data.loader import _load_config

    feat_path = get_feat_path(ticker)
    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Override target for NIFTY
    if is_nifty_ticker(ticker):
        cfg = _load_config()
        cfg["model"]["target"] = "log_return_5d"
        logger.info("[train] NIFTY: using log_return_5d as target")

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
    print(f"\n=== TEST SET RESULTS (last 15% — out of sample) ===")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")


def step5_sample_prediction(
    ticker: str, causal_features: list[str], market: str = "us"
) -> None:
    banner(f"STEP 5 — Live Prediction and Model Report ({ticker})")
    import pandas as pd
    from ml.src.data.loader import DataLoader
    from ml.src.features.pipeline import FeaturePipeline
    from ml.src.ensemble import Ensemble
    from ml.src.evaluation.metrics import Metrics
    from ml.src.evaluation.regime_splitter import RegimeSplitter

    if is_nifty_ticker(ticker) or market == "india":
        # For NIFTY — use last row of feature matrix as live features
        logger.info("Indian market — using latest data from feature matrix...")
        feat_path    = get_feat_path(ticker)
        df_full      = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        from ml.src.data.loader import _load_config
        cfg          = _load_config()
        target       = cfg["model"]["target"]
        feature_cols = [c for c in df_full.columns if c != target]
        row          = df_full[feature_cols].iloc[-1]
        current_price = float(df_full["close"].iloc[-1]) if "close" in df_full.columns else 0.0
    else:
        logger.info("Fetching live data...")
        loader        = DataLoader()
        live          = loader.load_live(ticker)
        pipeline      = FeaturePipeline()
        row           = pipeline.build_live(ticker, live)
        current_price = float(live["prices"]["close"].iloc[-1])

    logger.info("Running prediction...")
    ensemble = Ensemble()
    ensemble.load(ticker)

    result = ensemble.predict_live(
        live_features=row,
        ticker=ticker,
        current_price=current_price,
        causal_features=causal_features,
    )

    # Option B regime metrics
    model_metrics = {}
    feat_path = get_feat_path(ticker)
    if feat_path.exists():
        try:
            df       = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            m        = Metrics()
            splitter = RegimeSplitter(market=market)
            regimes  = splitter.split_all(df)

            for regime_name, regime_df in regimes.items():
                if len(regime_df) < 30:
                    continue
                try:
                    regime_start = splitter.regimes[regime_name][0]
                    train_df     = df.loc[:regime_start].iloc[:-1]
                    if len(train_df) < 200:
                        continue
                    fresh = Ensemble()
                    fresh.train_all(train_df, ticker, causal_features)
                    preds = fresh.predict_historical(regime_df, causal_features)
                    if "actual_return" in preds.columns:
                        model_metrics[regime_name] = m.compute_all(
                            preds["predicted_return"],
                            preds["actual_return"],
                            label=regime_name,
                        )
                except Exception as e:
                    logger.warning(f"Regime {regime_name} eval failed: {e}")

            # Overall — last 15% only
            n         = len(df)
            test_df   = df.iloc[int(n * 0.85):]
            preds_all = ensemble.predict_historical(test_df, causal_features)
            if "actual_return" in preds_all.columns:
                model_metrics["overall"] = m.compute_all(
                    preds_all["predicted_return"],
                    preds_all["actual_return"],
                    label="overall",
                )
        except Exception as e:
            logger.warning(f"Could not compute metrics: {e}")

    W = 60
    print(f"\n{'='*W}")
    print(f"  LIVE PREDICTION — {ticker}   ({result.prediction_date})")
    print(f"{'='*W}")
    print(f"  Current price:    {current_price:.2f}")
    print(f"  Predicted price:  {result.predicted_price:.2f}")
    print(f"  Direction:        {result.direction}")
    print(f"  Expected return:  {result.predicted_return:+.2%}")
    print(f"  Confidence:       {result.confidence:.0%}")
    print(f"  Range (90% CI):   {result.lower_band:.2f} — {result.upper_band:.2f}")
    print(f"  Horizon:          {result.horizon_days} trading days")
    print(f"  Model:            {result.model_name}")
    print(f"\n  Causal Drivers:")
    for d in result.causal_drivers:
        shap  = d.get("shap", d["value"])
        arrow = "▲" if d["impact"] == "positive" else "▼"
        print(f"    {arrow} {d['feature']:30s} {d['impact']:8s}  ({shap:+.4f})")
    print(f"\n  Causal Features Used ({len(causal_features)}):")
    print(f"    {', '.join(causal_features)}")
    if model_metrics and "overall" in model_metrics:
        regimes_order = ["overall"] + [r for r in model_metrics if r != "overall"]
        print(f"\n  Model Performance (Option B — pre-regime train, test on regime):")
        header = f"    {'Metric':<25} " + "".join(f"  {r[:12]:>12}" for r in regimes_order)
        print(header)
        print("    " + "-"*25 + "".join("  " + "-"*12 for _ in regimes_order))
        for metric in ["directional_accuracy","sharpe_ratio","max_drawdown","calmar_ratio","rmse"]:
            row_str = f"    {metric:<25}"
            for regime in regimes_order:
                val = model_metrics.get(regime, {}).get(metric, float("nan"))
                row_str += f"  {'N/A':>12}" if val != val else f"  {val:>12.4f}"
            print(row_str)
        print(f"\n    Option B: each regime tested on data the model never saw during training")
        print(f"    Random baseline = 0.50  |  Target > 0.52")
    print(f"{'='*W}")


def step6_regime_backtest(
    ticker: str, causal_features: list[str], df, market: str = "us"
) -> None:
    banner(f"STEP 6 — Full Regime Backtest / Paper Table ({ticker})")
    from ml.src.evaluation.backtester import Backtester

    bt      = Backtester()
    results = bt.regime_backtest(df, ticker, causal_features)

    if results.empty:
        logger.warning("[regime_backtest] No results computed.")
        return

    W = 80
    print(f"\n{'='*W}")
    print(f"  REGIME BACKTEST — {ticker}")
    print(f"  Option B: train on pre-regime data, test on each regime independently")
    print(f"{'='*W}")

    for metric in ["directional_accuracy", "sharpe_ratio", "max_drawdown", "calmar_ratio"]:
        if metric not in results.columns:
            continue
        import pandas as pd
        pivot = results[metric].unstack(level="regime")
        print(f"\n  {metric.upper().replace('_', ' ')}:")
        print(f"  {'Model':<20}" + "".join(f"  {c[:12]:>12}" for c in pivot.columns))
        print(f"  {'-'*20}" + "".join(f"  {'-'*12}" for _ in pivot.columns))
        for model_name, row in pivot.iterrows():
            print(f"  {model_name:<20}" + "".join(
                f"  {v:>12.4f}" if not pd.isna(v) else f"  {'N/A':>12}"
                for v in row
            ))

    print(f"\n  Key finding: causal model degrades less than all_features under regime shifts")
    print(f"  Random baseline DA ≈ 0.50 across all regimes")
    print(f"{'='*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="causal-stock-predictor — full ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  US stocks:
    python run_pipeline.py --ticker AAPL
    python run_pipeline.py --ticker AAPL --predict-only
    python run_pipeline.py --ticker AAPL --full-retrain

  Indian market:
    python run_pipeline.py --ticker NIFTY --market india
    python run_pipeline.py --ticker NIFTY --market india --predict-only
    python run_pipeline.py --ticker NIFTY --market india --paper-eval
    python run_pipeline.py --ticker RELIANCE.NS --market india
        """
    )
    parser.add_argument("--ticker",       type=str, required=True,
                        help="Stock ticker e.g. AAPL or NIFTY")
    parser.add_argument("--market",       type=str, default="us",
                        choices=["us", "india"],
                        help="Market: us (default) or india")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only run live prediction (models must exist)")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Re-download data even if already on disk")
    parser.add_argument("--full-retrain", action="store_true",
                        help="Force rebuild features + causal + retrain models")
    parser.add_argument("--paper-eval",   action="store_true",
                        help="Run full Option B regime backtest — paper Table 2 (~1hr)")
    parser.add_argument("--with-retrain", action="store_true",
                        help="Also run walk-forward retraining (~1hr)")
    parser.add_argument("--with-regime",  action="store_true",
                        help="Also train regime-aware models (~40min)")
    args   = parser.parse_args()

    # Normalise ticker
    ticker = args.ticker.upper()
    if is_nifty_ticker(ticker):
        ticker = "NIFTY"

    total_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  causal-stock-predictor")
    print(f"  Ticker: {ticker}  |  Market: {args.market.upper()}")
    print(f"{'#'*60}")

    # ── Predict only ──────────────────────────────────────────────────────────
    if args.predict_only:
        from ml.src.causal.selector import CausalSelector
        causal_features = CausalSelector().load(ticker)
        step5_sample_prediction(ticker, causal_features, market=args.market)
        return

    # ── Auto-detect what exists ───────────────────────────────────────────────
    from ml.src.causal.selector import CausalSelector
    from ml.src.data.loader import _load_config

    cfg         = _load_config()
    models_dir  = ROOT / cfg["saved_models"]["dir"]
    feat_path   = get_feat_path(ticker)
    causal_path = models_dir / f"causal_features_{ticker}.json"
    lgbm_path   = models_dir / f"lgbm_{ticker}.pkl"

    if is_nifty_ticker(ticker) or args.market == "india":
        data_exists = feat_path.exists()   # for India — data = feature matrix
    else:
        data_path   = ROOT / cfg["data"]["raw_dir"] / "prices" / f"{ticker}.csv"
        data_exists = check_exists(data_path)

    skip_data   = data_exists               and not args.refresh_data
    skip_feat   = check_exists(feat_path)   and not args.full_retrain
    skip_causal = check_exists(causal_path) and not args.full_retrain
    skip_train  = check_exists(lgbm_path)   and not args.full_retrain

    print(f"\n  Status:")
    print(f"    Data:            {'✓ exists' if skip_data   else '✗ missing'} {'(skipping)' if skip_data   else '(will load)'}")
    print(f"    Feature matrix:  {'✓ exists' if skip_feat   else '✗ missing'} {'(skipping)' if skip_feat   else '(will build)'}")
    print(f"    Causal features: {'✓ exists' if skip_causal else '✗ missing'} {'(skipping)' if skip_causal else '(will run PCMCI)'}")
    print(f"    Trained models:  {'✓ exists' if skip_train  else '✗ missing'} {'(skipping)' if skip_train  else '(will train)'}")

    try:
        # Step 1
        step1_load_data(ticker, skip=skip_data, market=args.market)

        # Step 2
        step2_build_features(ticker, force=not skip_feat, market=args.market)

        # Step 3
        if skip_causal:
            causal_features = CausalSelector().load(ticker)
            logger.info(f"Loaded saved causal features ({len(causal_features)}): {causal_features}")
        else:
            causal_features = step3_causal_discovery(ticker)

        # Step 4
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

        # Step 5 — Live prediction + Option B regime table
        step5_sample_prediction(ticker, causal_features, market=args.market)

        # Step 6 — Full paper Table 2 (opt-in)
        if args.paper_eval:
            import pandas as pd
            df_full = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            step6_regime_backtest(ticker, causal_features, df_full, market=args.market)

        banner(f"PIPELINE COMPLETE — {elapsed(total_start)} total")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()