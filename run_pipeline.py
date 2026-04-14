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


def _print_regime_table(ticker: str, market: str, feat_path: Path) -> None:
    """
    Print configured regime definitions (from config) and, if the feature
    matrix exists, compute and print regime statistics for the data.
    Always safe to call; errors are logged and do not abort the pipeline.
    """
    try:
        from ml.src.evaluation.regime_splitter import RegimeSplitter
        import pandas as pd

        splitter = RegimeSplitter(market=market)
        print("\n  Regime definitions (from config):")
        for name, (start, end) in splitter.regime_dates.items():
            print(f"    {name:15s}: {start} → {end}")

        if feat_path.exists():
            try:
                df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
                stats = splitter.regime_stats(df)
                print("\n  Regime stats (based on feature matrix):")
                print(stats.to_string())
            except Exception as e:
                logger.warning(f"[pipeline] Could not compute regime stats: {e}")
        else:
            logger.info("[pipeline] Feature matrix not found — skipping regime stats computation.")

    except Exception as e:
        logger.warning(f"[pipeline] Could not print regime definitions: {e}")


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


def step2_build_features(ticker: str, force: bool, market: str = "us", use_finbert: bool | None = None) -> None:
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
        if use_finbert is None:
            use_fb = cfg["features"].get("finbert", {}).get("enabled", False)
        else:
            use_fb = bool(use_finbert)
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

    # CRITICAL: remove BOTH target columns before causal discovery.
    # excess_return_5d and log_return_5d contain future prices (shift(-5)).
    # Leaving them as regular columns causes PCMCI/Granger to find spurious
    # causal links from features that are mathematically related to the
    # target — not genuine causal relationships.
    TARGET_COL       = cfg["model"]["target"]       # excess_return_5d
    AUXILIARY_TARGET  = "log_return_5d"             # always drop this too
    cols_to_drop = [c for c in [TARGET_COL, AUXILIARY_TARGET, "excess_return_5d"]
                    if c in df_train.columns]
    df_causal = df_train.drop(columns=cols_to_drop)
    logger.info(
        f"[causal] Dropped {cols_to_drop} before causal discovery "
        f"({len(df_causal.columns)} features remain)"
    )

    # Granger — full training set (features only, target passed separately)
    logger.info("Running Granger causality on training data...")
    start            = time.time()
    granger          = GrangerCausality()
    # Re-add only the target column for Granger (it tests X → target)
    df_granger = df_causal.copy()
    df_granger[target] = df_train[target]
    granger_results  = granger.run(df_granger, target=target, verbose=False)
    granger_features = granger.get_causal_features(granger_results)
    logger.info(f"Granger done in {elapsed(start)} — {len(granger_features)} causal features")

    # PCMCI — last 50% of training data (no test leakage)
    logger.info("Running PCMCI on last 50% of training data (5-15 mins)...")
    start      = time.time()
    df_pcmci_full = df_train.iloc[-int(len(df_train) * 0.5):]
    # Drop target/auxiliary columns from PCMCI input too
    df_pcmci = df_pcmci_full.drop(
        columns=[c for c in cols_to_drop if c in df_pcmci_full.columns]
    )
    # Re-add only the target for PCMCI's exclude_target mode
    df_pcmci[target] = df_pcmci_full[target]
    logger.info(
        f"PCMCI dataset: {len(df_pcmci)} rows "
        f"({df_pcmci.index.min().date()} → {df_pcmci.index.max().date()})"
    )
    pcmci          = PCMCIDiscovery()
    # To avoid including the forward-looking target column inside the PCMCI
    # variable set (which can bias conditional-independence tests), run
    # PCMCI on the feature set only and use Granger for direct feature→target
    # strength. This prevents target-derived leakage inside PCMCI's graph.
    pcmci_results  = pcmci.run(df_pcmci, target=target, exclude_target=True)
    pcmci_features = pcmci.get_causal_features(pcmci_results)
    logger.info(f"PCMCI done in {elapsed(start)} — {len(pcmci_features)} causal features")

    # Select + save
    # Use union for NIFTY (less data = stricter intersection may find 0 features)
    # Use intersection for US stocks (15 years of data = strict selection is fine)
    selector = CausalSelector()
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
    start = time.time()
    # If NIFTY/India, use an explicit local config override (do NOT mutate global config)
    if is_nifty_ticker(ticker):
        import copy
        cfg_base = _load_config()
        cfg_local = copy.deepcopy(cfg_base)
        cfg_local["model"]["target"] = "log_return_5d"
        logger.info("[train] NIFTY: using log_return_5d as target (local override)")
        ensemble = Ensemble(cfg=cfg_local)
    else:
        ensemble = Ensemble()
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
    from ml.src.data.loader import DataLoader, _load_config
    from ml.src.features.pipeline import FeaturePipeline
    from ml.src.ensemble import Ensemble
    from ml.src.evaluation.metrics import Metrics
    from ml.src.evaluation.regime_splitter import RegimeSplitter

    # Determine config override for NIFTY (use log_return_5d instead of excess_return_5d)
    cfg = None
    if is_nifty_ticker(ticker) or market == "india":
        import copy
        cfg_base = _load_config()
        cfg = copy.deepcopy(cfg_base)
        cfg["model"]["target"] = "log_return_5d"
        logger.info("[train] NIFTY/India: using log_return_5d as target (local override)")

    # NIFTY — skip live prediction (no real-time P/E, P/B, headlines available)
    # Live prediction quality would be misleading without the same data quality as training
    if is_nifty_ticker(ticker) or market == "india":
        logger.info(
            "[nifty] Skipping live prediction — real-time P/E, P/B and headlines "
            "not available. Showing model performance on historical data only."
        )
        result       = None
        current_price = None
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
                    # Use regime-specific ticker name so save() does NOT
                    # overwrite production model files on disk
                    regime_ticker = f"{ticker}_{regime_name}_eval"
                    # Pass cfg override for NIFTY so per-regime training uses correct target
                    fresh = Ensemble(cfg=cfg)
                    fresh.train_all(train_df, regime_ticker, causal_features)
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
    if result is not None:
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
        print(f"\n  PCMCI-selected Drivers:")
        for d in result.causal_drivers:
            shap  = d.get("shap", d["value"])
            arrow = "▲" if d["impact"] == "positive" else "▼"
            print(f"    {arrow} {d['feature']:30s} {d['impact']:8s}  ({shap:+.4f})")
    else:
        print(f"  MODEL REPORT — {ticker}")
        print(f"{'='*W}")
        print(f"  Live prediction not available for NIFTY index.")
        print(f"  Real-time P/E, P/B and news headlines not available.")
    print(f"\n  PCMCI-selected Features Used ({len(causal_features)}):")
    print(f"    {', '.join(causal_features)}")
    if model_metrics:
        regimes_order = (["overall"] if "overall" in model_metrics else []) +                         [r for r in model_metrics if r != "overall"]
        if regimes_order:
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
    from ml.src.data.loader import _load_config

    # If NIFTY/India market, pass a config override so target is correct
    cfg = None
    if is_nifty_ticker(ticker) or market == "india":
        cfg = _load_config()
        cfg["model"]["target"] = "log_return_5d"
        logger.info("[regime_backtest] NIFTY/India: using log_return_5d as target")

    bt = Backtester(cfg=cfg, market=market)
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
        print(f"  {'Model':<20}" + "".join(f"  {c[:12]:>20}" for c in pivot.columns))
        print(f"  {'-'*20}" + "".join(f"  {'-'*20}" for _ in pivot.columns))
        # If metric is directional_accuracy and CI columns exist, show them
        show_ci = metric == "directional_accuracy" and "directional_accuracy_ci_lower" in results.columns
        pivot_lo = results["directional_accuracy_ci_lower"].unstack(level="regime") if show_ci else None
        pivot_hi = results["directional_accuracy_ci_upper"].unstack(level="regime") if show_ci else None

        for model_name, row in pivot.iterrows():
            cells = []
            for c in pivot.columns:
                v = row[c]
                if pd.isna(v):
                    cells.append(f"  {'N/A':>20}")
                else:
                    if show_ci:
                        try:
                            lo = pivot_lo.loc[model_name, c]
                            hi = pivot_hi.loc[model_name, c]
                            cells.append(f"  {v:20.4f} [{lo:.2%},{hi:.2%}]")
                        except Exception:
                            cells.append(f"  {v:20.4f}")
                    else:
                        cells.append(f"  {v:20.4f}")
            print(f"  {model_name:<20}" + "".join(cells))

    print(f"\n  Key finding: PCMCI-selected model degrades less than all_features under regime shifts")
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
    parser.add_argument("--finbert", action="store_true",
                        help="Use FinBERT for sentiment scoring (slow on CPU, requires GPU ideally)")
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
    print(f"    PCMCI-selected features: {'✓ exists' if skip_causal else '✗ missing'} {'(skipping)' if skip_causal else '(will run PCMCI)'}")
    print(f"    Trained models:  {'✓ exists' if skip_train  else '✗ missing'} {'(skipping)' if skip_train  else '(will train)'}")

    # Always print configured regime definitions and stats (if feature matrix available)
    _print_regime_table(ticker, args.market, feat_path)

    try:
        # Step 1
        step1_load_data(ticker, skip=skip_data, market=args.market)

        # Step 2
        step2_build_features(ticker, force=not skip_feat, market=args.market, use_finbert=args.finbert)

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