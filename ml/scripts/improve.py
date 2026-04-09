"""
improve.py
----------
Small CLI to run the improvement scaffolding: diagnostics, quick-wins, ...

Examples:
    python ml/scripts/improve.py baseline --ticker NIFTY --market india
    python ml/scripts/improve.py tune-threshold --ticker NIFTY --market india
"""

import argparse
import logging
import pandas as pd

from ml.src.improvements.diagnostics import run_baseline
from ml.src.improvements.quick_wins import tune_threshold
from ml.src.data.validator import DataValidator
from ml.src.data.loader import _load_config
from ml.src.improvements.cleaning import clean_feature_matrix
from ml.src.improvements.feature_augment import augment_features
from ml.src.improvements.target_transform import transform_target
from ml.src.improvements.hpo import run_hpo
from ml.src.models.regime_model import RegimeAwareEnsemble
from ml.src.causal.selector import CausalSelector
from ml.src.evaluation.backtester import Backtester
from ml.src.improvements.monitor import monitor_model
from ml.src.improvements.meta_tune import train_meta_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("improve")


def main():
    parser = argparse.ArgumentParser(description="Run improvement utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("baseline", help="Run baseline diagnostics")
    p1.add_argument("--ticker", required=True)
    p1.add_argument("--market", choices=["us", "india"], default="us")
    p1.add_argument("--force-retrain", action="store_true")

    pv = sub.add_parser("validate", help="Run data validation on feature matrix")
    pv.add_argument("--ticker", required=True)
    pv.add_argument("--market", choices=["us", "india"], default="us")

    pc = sub.add_parser("clean", help="Clean feature matrix and save cleaned copy")
    pc.add_argument("--ticker", required=True)
    pc.add_argument("--market", choices=["us", "india"], default="us")
    pc.add_argument("--drop-nan-ratio", type=float, default=0.01)

    pa = sub.add_parser("augment", help="Augment feature matrix (lags, vol, momentum)")
    pa.add_argument("--ticker", required=True)
    pa.add_argument("--market", choices=["us", "india"], default="us")

    ph = sub.add_parser("hpo", help="Run hyperparameter tuning (Optuna) for a ticker")
    ph.add_argument("--ticker", required=True)
    ph.add_argument("--market", choices=["us", "india"], default="us")
    ph.add_argument("--trials", type=int, default=50)

    pr = sub.add_parser("train-regimes", help="Train regime-specific models for a ticker")
    pr.add_argument("--ticker", required=True)
    pr.add_argument("--market", choices=["us", "india"], default="us")
    pr.add_argument("--min-samples", type=int, default=200)

    pm = sub.add_parser("meta-tune", help="Train classification meta-learner optimised for direction")
    pm.add_argument("--ticker", required=True)
    pm.add_argument("--market", choices=["us", "india"], default="us")

    pb = sub.add_parser("backtest", help="Run backtesting utilities")
    pb.add_argument("--ticker", required=True)
    pb.add_argument("--market", choices=["us", "india"], default="us")
    pb.add_argument("--mode", choices=["regime", "walk", "test"], default="regime")

    pmn = sub.add_parser("monitor", help="Run lightweight monitoring and fallback")
    pmn.add_argument("--ticker", required=True)
    pmn.add_argument("--market", choices=["us", "india"], default="us")
    pmn.add_argument("--threshold", type=float, default=0.60)

    pt = sub.add_parser("transform-target", help="Transform target: smoothing / binary label / discretize")
    pt.add_argument("--ticker", required=True)
    pt.add_argument("--market", choices=["us", "india"], default="us")
    pt.add_argument("--smoothing-window", type=int, default=None)
    pt.add_argument("--threshold", type=float, default=0.0)
    pt.add_argument("--discretize-quantiles", type=int, default=None)

    p2 = sub.add_parser("tune-threshold", help="Tune threshold quick-win")
    p2.add_argument("--ticker", required=True)
    p2.add_argument("--market", choices=["us", "india"], default="us")
    p2.add_argument("--low", type=float, default=-0.02)
    p2.add_argument("--high", type=float, default=0.02)
    p2.add_argument("--n", type=int, default=81)

    args = parser.parse_args()

    if args.cmd == "baseline":
        run_baseline(args.ticker, market=args.market, force_retrain=args.force_retrain)
    elif args.cmd == "tune-threshold":
        tune_threshold(args.ticker, market=args.market, low=args.low, high=args.high, n=args.n)
    elif args.cmd == "validate":
        cfg = _load_config()
        # locate feature matrix
        if args.market == "india" or args.ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50"):
            from ml.src.data.nifty_loader import NiftyLoader
            feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        else:
            pipeline = __import__("ml.src.features.pipeline", fromlist=["FeaturePipeline"]).FeaturePipeline()
            feat_path = pipeline.features_dir / f"{args.ticker.upper()}_features.csv"

        if not feat_path.exists():
            print(f"ERROR: No feature matrix at {feat_path}")
            return

        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        validator = DataValidator(strict=False)
        report = validator.validate_feature_matrix(df, name=f"{args.ticker.upper()} features")
        print(report)
    elif args.cmd == "clean":
        report = clean_feature_matrix(args.ticker, market=args.market, drop_nan_ratio=args.drop_nan_ratio)
        print(report)
    elif args.cmd == "augment":
        report = augment_features(args.ticker, market=args.market)
        print(report)
    elif args.cmd == "hpo":
        report = run_hpo(args.ticker, market=args.market, n_trials=args.trials)
        print(report)
    elif args.cmd == "train-regimes":
        # load feature matrix
        if args.market == "india" or args.ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50"):
            from ml.src.data.nifty_loader import NiftyLoader
            feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        else:
            pipeline = __import__("ml.src.features.pipeline", fromlist=["FeaturePipeline"]).FeaturePipeline()
            feat_path = pipeline.features_dir / f"{args.ticker.upper()}_features.csv"

        if not feat_path.exists():
            print(f"ERROR: No feature matrix at {feat_path}")
            return

        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        try:
            causal_features = CausalSelector().load(args.ticker)
        except Exception:
            causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != _load_config()["model"]["target"]][:15]

        model = RegimeAwareEnsemble(cfg=_load_config())
        results = model.fit_all_regimes(df, args.ticker, causal_features, min_samples=args.min_samples)
        print(results)
    elif args.cmd == "meta-tune":
        report = train_meta_classifier(args.ticker, market=args.market)
        print(report)
    elif args.cmd == "backtest":
        # load feature matrix
        if args.market == "india" or args.ticker.upper() in ("NIFTY", "^NSEI", "NIFTY50"):
            from ml.src.data.nifty_loader import NiftyLoader
            feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        else:
            pipeline = __import__("ml.src.features.pipeline", fromlist=["FeaturePipeline"]).FeaturePipeline()
            feat_path = pipeline.features_dir / f"{args.ticker.upper()}_features.csv"

        if not feat_path.exists():
            print(f"ERROR: No feature matrix at {feat_path}")
            return

        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        bt = Backtester(market=args.market)
        try:
            causal_features = CausalSelector().load(args.ticker)
        except Exception:
            causal_features = [c for c in df.select_dtypes(include=["number"]).columns if c != _load_config()["model"]["target"]][:15]

        if args.mode == "regime":
            res = bt.regime_backtest(df, args.ticker, causal_features)
            print(res)
        elif args.mode == "walk":
            res = bt.walk_forward(df, args.ticker, causal_features)
            print(res)
        else:
            res = bt.evaluate_test_set(args.ticker, causal_features, df)
            print(res)
    elif args.cmd == "monitor":
        report = monitor_model(args.ticker, market=args.market, threshold=args.threshold)
        print(report)
    elif args.cmd == "transform-target":
        report = transform_target(
            args.ticker,
            market=args.market,
            smoothing_window=args.smoothing_window,
            threshold=args.threshold,
            discretize_quantiles=args.discretize_quantiles,
        )
        print(report)


if __name__ == "__main__":
    main()
