"""
run_paper_validation.py
-----------------------
Single script that runs ALL paper-critical validation checks.

This version uses a runtime adapter (ensemble_predict) to handle
different Ensemble interface versions — it introspects the actual
object rather than hardcoding 'predict_historical'.

Usage:
    python run_paper_validation.py --ticker NIFTY --market india --quick
    python run_paper_validation.py --ticker AAPL
    python run_paper_validation.py --ticker AAPL --full
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("paper_validation")


def banner(msg: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def elapsed(start: float) -> str:
    s = time.time() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}min"


# ---------------------------------------------------------------------------
# Ensemble compatibility adapter
# ---------------------------------------------------------------------------

def ensemble_predict(ensemble, df, causal_features):
    """
    Runtime-adaptive prediction wrapper.

    The 'new' branch Ensemble may have a different method name than
    predict_historical(). This function tries known method names in
    order, then normalises the output to a standard DataFrame with
    columns: predicted_return, actual_return.
    """
    import pandas as pd
    import numpy as np
    from ml.src.data.loader import _load_config

    cfg        = _load_config()
    target_col = cfg["model"]["target"]

    # Try method names in priority order
    method_candidates = [
        "predict_historical",
        "predict_batch",
        "predict",
        "backtest",
        "score",
        "evaluate",
    ]

    predict_fn  = None
    method_used = None
    for name in method_candidates:
        if hasattr(ensemble, name) and callable(getattr(ensemble, name)):
            predict_fn  = getattr(ensemble, name)
            method_used = name
            break

    if predict_fn is None:
        all_methods = [m for m in dir(ensemble)
                       if callable(getattr(ensemble, m)) and not m.startswith("_")]
        raise AttributeError(
            f"Ensemble has no recognized predict method. "
            f"Available public methods: {all_methods}"
        )

    logger.info(f"[adapter] Using Ensemble.{method_used}()")

    # Filter to features that exist in df
    feat_cols = [c for c in causal_features if c in df.columns]
    if not feat_cols:
        raise ValueError(
            f"None of the causal features are in the DataFrame. "
            f"First 5 features: {causal_features[:5]}, "
            f"First 5 df cols: {list(df.columns[:5])}"
        )

    # Try calling with progressively simpler signatures
    raw = None
    for call_args in [
        (df, feat_cols),          # predict_historical(df, causal_features)
        (df,),                    # predict_historical(df)
        (df[feat_cols],),         # predict(X)
        (df[feat_cols], feat_cols), # predict(X, features)
    ]:
        try:
            raw = predict_fn(*call_args)
            break
        except TypeError:
            continue
        except Exception as e:
            raise RuntimeError(
                f"[adapter] {method_used}{call_args} raised: {e}"
            )

    if raw is None:
        raise RuntimeError(
            f"[adapter] All call signatures failed for {method_used}()"
        )

    # Normalise output to standard DataFrame
    if isinstance(raw, pd.DataFrame):
        result = raw.copy()

        # Rename columns to standard names if needed
        col_renames = {}
        for col in result.columns:
            cl = col.lower()
            if col != "predicted_return" and "predict" in cl and "return" in cl:
                col_renames[col] = "predicted_return"
            elif col != "predicted_return" and cl in ("pred", "prediction", "forecast", "y_pred"):
                col_renames[col] = "predicted_return"
            elif col != "actual_return" and cl in ("actual", "y_true", "true_return", "label"):
                col_renames[col] = "actual_return"
        if col_renames:
            result = result.rename(columns=col_renames)
            logger.info(f"[adapter] Renamed columns: {col_renames}")

        # Add actual_return from df if missing
        if "actual_return" not in result.columns and target_col in df.columns:
            result["actual_return"] = df[target_col].reindex(result.index)

        return result

    elif isinstance(raw, (np.ndarray, pd.Series)):
        if isinstance(raw, np.ndarray):
            idx = df.index[:len(raw)]
            raw = pd.Series(raw, index=idx)
        result = pd.DataFrame({"predicted_return": raw}, index=raw.index)
        if target_col in df.columns:
            result["actual_return"] = df[target_col].reindex(raw.index)
        return result

    else:
        raise TypeError(
            f"[adapter] {method_used}() returned unexpected type: {type(raw)}"
        )


def load_ensemble(ticker: str):
    """Load ensemble, return (ensemble, loaded_ok)."""
    from ml.src.ensemble import Ensemble
    ensemble = Ensemble()
    try:
        ensemble.load(ticker)
        return ensemble, True
    except Exception as e:
        logger.warning(f"Could not load ensemble for {ticker}: {e}")
        return ensemble, False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(ticker: str, market: str):
    import pandas as pd
    from ml.src.data.loader import _load_config
    from ml.src.causal.selector import CausalSelector

    cfg = _load_config()
    if market == "india" or ticker in ("NIFTY", "^NSEI"):
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        target    = "log_return_5d"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"
        target    = cfg["model"]["target"]

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {feat_path}. "
            f"Run run_pipeline.py --ticker {ticker} first."
        )

    df        = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    train_end = int(len(df) * cfg["model"]["train_ratio"])
    df_train  = df.iloc[:train_end]

    try:
        causal_features = CausalSelector().load(ticker)
    except FileNotFoundError:
        logger.warning("No causal features saved — using numeric columns as fallback")
        causal_features = [
            c for c in df.select_dtypes(include=["float"]).columns
            if c != target
        ][:15]

    return df, df_train, causal_features, target, cfg


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_1_pcmci_stability(df_train, target, ticker, quick, results):
    banner("CHECK 1: PCMCI Feature Stability")
    if quick:
        logger.info("Skipping PCMCI stability (--quick mode)")
        results["pcmci_stability"] = {"skipped": True}
        return

    try:
        from ml.src.causal.stability import PCMCIStabilityAnalyzer
    except ImportError:
        logger.warning("stability.py not found — copy from corrections/ml/src/causal/")
        results["pcmci_stability"] = {"skipped": True, "reason": "not installed"}
        return

    start    = time.time()
    analyzer = PCMCIStabilityAnalyzer(n_windows=3)
    report   = analyzer.run(df_train, target=target, ticker=ticker)

    results["pcmci_stability"] = {
        "mean_jaccard":         report.mean_jaccard,
        "std_jaccard":          report.std_jaccard,
        "verdict":              report.verdict,
        "stable_core":          sorted(report.stable_core),
        "majority_core":        sorted(report.majority_core),
        "recommended_strategy": report.recommended_strategy(),
        "elapsed":              elapsed(start),
    }

    if report.verdict == "UNSTABLE":
        logger.warning(
            f"⚠ PCMCI UNSTABLE (Jaccard={report.mean_jaccard:.3f}). "
            f"Use 'union' strategy and disclose in paper."
        )
    else:
        logger.info(
            f"✓ PCMCI {report.verdict} (Jaccard={report.mean_jaccard:.3f})"
        )


def check_2_nifty_atr(df, market, ticker, results):
    banner("CHECK 2: ATR Diagnostic (Nifty H/L contamination)")
    if market != "india" and ticker not in ("NIFTY", "^NSEI"):
        logger.info("Not Nifty — skipping.")
        results["atr_diagnostic"] = {"skipped": True, "reason": "not nifty"}
        return

    try:
        from ml.src.features.nifty_feature_guard import NiftyFeatureGuard
    except ImportError:
        logger.warning("nifty_feature_guard.py not found — copy from corrections/")
        results["atr_diagnostic"] = {"skipped": True, "reason": "not installed"}
        return

    guard = NiftyFeatureGuard()
    diag  = guard.print_atr_diagnostic(df)
    results["atr_diagnostic"] = {
        k: float(v) if hasattr(v, "__float__") else v
        for k, v in diag.items()
    }

    corr = diag.get("atr_close_correlation", 0)
    if corr > 0.8:
        logger.warning(f"⚠ ATR/Close corr={corr:.4f} — exclude atr_14 from causal discovery.")
    else:
        logger.info(f"✓ ATR/Close corr={corr:.4f} — acceptable.")


def check_3_significance(df, causal_features, target, ticker, market, results):
    banner("CHECK 3: Statistical Significance Tests")

    try:
        from ml.src.evaluation.significance import SignificanceTester
    except ImportError:
        logger.warning("significance.py not found — copy from corrections/ml/src/evaluation/")
        results["significance"] = {"skipped": True, "reason": "not installed"}
        return

    ensemble, loaded = load_ensemble(ticker)
    if not loaded:
        logger.warning("[check_3] Training ensemble on 85% split...")
        try:
            n = len(df)
            ensemble.train_all(df.iloc[:int(n * 0.85)], ticker, causal_features)
        except Exception as e:
            results["significance"] = {"error": f"Ensemble unavailable: {e}"}
            return

    n       = len(df)
    test_df = df.iloc[int(n * 0.85):]

    try:
        preds = ensemble_predict(ensemble, test_df, causal_features)
    except Exception as e:
        results["significance"] = {"error": f"Prediction failed: {e}"}
        logger.error(f"[check_3] {e}")
        return

    if "actual_return" not in preds.columns or preds["actual_return"].isna().all():
        results["significance"] = {"error": "actual_return missing or all-NaN"}
        return

    y_pred = preds["predicted_return"].dropna()
    y_true = preds["actual_return"].reindex(y_pred.index).dropna()
    y_pred = y_pred.reindex(y_true.index)

    if len(y_true) < 30:
        results["significance"] = {"error": f"Too few test rows: {len(y_true)}"}
        return

    tester = SignificanceTester()
    binom  = tester.binomial_test(y_pred, y_true)
    boot   = tester.bootstrap_da_ci(y_pred, y_true, n_bootstrap=1000)

    print(f"\n{binom}")
    print(f"\n{boot}")

    results["significance"] = {
        "test_set_n":           int(len(y_true)),
        "test_set_da":          float(binom.statistic),
        "binomial_p":           float(binom.p_value),
        "binomial_significant": bool(binom.significant),
        "bootstrap_ci_low":     float(boot.ci_low),
        "bootstrap_ci_high":    float(boot.ci_high),
        "bootstrap_p":          float(boot.p_value),
        "ci_above_random":      bool(boot.significant),
    }

    if binom.significant:
        logger.info(
            f"✓ DA={binom.statistic:.3f} significantly > 0.50 "
            f"(p={binom.p_value:.4f}, n={len(y_true)})"
        )
    else:
        logger.warning(
            f"⚠ DA={binom.statistic:.3f} NOT significantly > 0.50 "
            f"(p={binom.p_value:.4f}) — critical paper weakness."
        )


def check_4_arima_variance(df, causal_features, target, ticker, results):
    banner("CHECK 4: ARIMA Prediction Variance")

    try:
        from ml.src.models.arima_model import ARIMAModel
    except ImportError:
        results["arima_variance"] = {"skipped": True, "reason": "ARIMAModel not importable"}
        return

    n        = len(df)
    train_df = df.iloc[:int(n * 0.70)]
    val_df   = df.iloc[int(n * 0.70):int(n * 0.85)]

    feat_col = next((c for c in causal_features if c in df.columns), df.columns[0])
    X_train  = train_df[[feat_col]]
    X_val    = val_df[[feat_col]]
    y_train  = train_df[target]
    y_val    = val_df[target]

    try:
        model = ARIMAModel()
        model.fit(X_train, y_train)
    except Exception as e:
        results["arima_variance"] = {"error": f"ARIMA fit failed: {e}"}
        return

    # predict_raw — should ideally be non-constant after correction
    raw_std  = None
    raw_const = None
    try:
        raw_preds = model.predict_raw(X_val)
        raw_std   = float(raw_preds.std())
        raw_const = bool(raw_std < 1e-10)
    except Exception as e:
        logger.warning(f"[check_4] predict_raw failed: {e}")

    # predict_val_set — only exists in corrected version
    val_std = None
    if hasattr(model, "predict_val_set"):
        try:
            val_preds = model.predict_val_set(y_val)
            val_std   = float(val_preds.std())
            logger.info(
                f"  predict_val_set std={val_std:.5f} "
                f"({'✓ varies' if val_std > 1e-10 else '⚠ constant'})"
            )
        except Exception as e:
            logger.warning(f"[check_4] predict_val_set failed: {e}")
    else:
        logger.warning(
            "[check_4] predict_val_set() not on ARIMAModel — "
            "install corrected arima_model.py from corrections/"
        )

    results["arima_variance"] = {
        "predict_raw_std":      raw_std,
        "predict_val_set_std":  val_std,
        "predict_raw_constant": raw_const,
        "has_predict_val_set":  hasattr(model, "predict_val_set"),
    }

    if raw_const:
        logger.warning(
            "⚠ ARIMA predict_raw CONSTANT — meta-learner ignores ARIMA. "
            "Install corrected arima_model.py."
        )
    elif raw_std is not None:
        logger.info(f"✓ ARIMA predict_raw std={raw_std:.5f}")


def check_5_calibration(df, causal_features, target, ticker, results):
    banner("CHECK 5: Confidence Calibration (ECE)")

    try:
        from ml.src.models.calibration import ConfidenceCalibrator
    except ImportError:
        logger.warning("calibration.py not found — copy from corrections/ml/src/models/")
        results["calibration"] = {"skipped": True, "reason": "not installed"}
        return

    ensemble, loaded = load_ensemble(ticker)
    if not loaded:
        results["calibration"] = {"skipped": True, "reason": "ensemble not loaded"}
        return

    n      = len(df)
    val_df = df.iloc[int(n * 0.70):int(n * 0.85)]

    try:
        val_preds = ensemble_predict(ensemble, val_df, causal_features)
    except Exception as e:
        results["calibration"] = {"skipped": True, "reason": f"prediction failed: {e}"}
        logger.warning(f"[check_5] {e}")
        return

    if "actual_return" not in val_preds.columns:
        results["calibration"] = {"skipped": True, "reason": "no actual_return"}
        return

    raw_conf = (val_preds["predicted_return"].abs() * 10).clip(0.3, 0.9)
    correct  = (
        (val_preds["predicted_return"] >= 0) ==
        (val_preds["actual_return"] >= 0)
    ).astype(float)

    mask     = raw_conf.notna() & correct.notna()
    raw_conf = raw_conf[mask].values
    correct  = correct[mask].values

    if len(raw_conf) < 30:
        results["calibration"] = {
            "skipped": True,
            "reason":  f"too few val samples ({len(raw_conf)})"
        }
        return

    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(raw_conf, correct)
    calibrator.print_reliability_table(raw_conf, correct)

    results["calibration"] = {
        "n_val_samples": int(len(raw_conf)),
        "mean_accuracy": float(correct.mean()),
    }
    logger.info(f"✓ Calibration done (n={len(raw_conf)}, acc={correct.mean():.3f})")


def check_8_sharpe(df, causal_features, target, ticker, results):
    banner("CHECK 8: Sharpe Ratio (original vs turnover-corrected)")

    try:
        from ml.src.evaluation.metrics import Metrics
        m = Metrics()
        if not hasattr(m, "sharpe_ratio_original"):
            logger.warning(
                "sharpe_ratio_original() not found — "
                "install corrected metrics.py from corrections/"
            )
            results["sharpe_comparison"] = {"skipped": True, "reason": "not installed"}
            return
    except ImportError:
        results["sharpe_comparison"] = {"skipped": True, "reason": "metrics not importable"}
        return

    ensemble, loaded = load_ensemble(ticker)
    if not loaded:
        results["sharpe_comparison"] = {"skipped": True, "reason": "ensemble not loaded"}
        return

    n       = len(df)
    test_df = df.iloc[int(n * 0.85):]

    try:
        preds = ensemble_predict(ensemble, test_df, causal_features)
    except Exception as e:
        results["sharpe_comparison"] = {"skipped": True, "reason": f"prediction failed: {e}"}
        return

    if "actual_return" not in preds.columns:
        results["sharpe_comparison"] = {"skipped": True, "reason": "no actual_return"}
        return

    y_pred = preds["predicted_return"].dropna()
    y_true = preds["actual_return"].reindex(y_pred.index).dropna()
    y_pred = y_pred.reindex(y_true.index)

    if len(y_true) < 20:
        results["sharpe_comparison"] = {
            "skipped": True,
            "reason":  f"too few test rows ({len(y_true)})"
        }
        return

    sr_orig   = m.sharpe_ratio_original(y_pred, y_true)
    sr_new    = m.sharpe_ratio(y_pred, y_true)
    sr_scaled = m.sharpe_ratio_scaled(y_pred, y_true)

    print(f"\n  Sharpe original (flat 10bps/period): {sr_orig:.4f}")
    print(f"  Sharpe corrected (turnover-aware):   {sr_new:.4f}")
    print(f"  Sharpe scaled (confidence-weighted): {sr_scaled:.4f}")

    results["sharpe_comparison"] = {
        "original_flat_cost":   float(sr_orig),
        "corrected_turnover":   float(sr_new),
        "scaled_by_confidence": float(sr_scaled),
        "n_test":               int(len(y_true)),
    }
    logger.info("✓ Sharpe comparison complete")


# ---------------------------------------------------------------------------
# Wrapper and main
# ---------------------------------------------------------------------------

def _run_check(name: str, fn, *args):
    """Run a check, catch all exceptions so others still run."""
    try:
        fn(*args)
    except Exception as e:
        logger.error(f"[{name}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def _print_summary(results: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  VALIDATION SUMMARY — {results.get('ticker', '?')}")
    print(f"{'='*65}")

    checks = [
        ("PCMCI stability",   results.get("pcmci_stability",   {})),
        ("ATR diagnostic",    results.get("atr_diagnostic",    {})),
        ("Significance",      results.get("significance",      {})),
        ("ARIMA variance",    results.get("arima_variance",    {})),
        ("Calibration",       results.get("calibration",       {})),
        ("Sharpe comparison", results.get("sharpe_comparison", {})),
    ]

    for name, check in checks:
        if not check:
            print(f"       ?  {name}: not run")
            continue
        if check.get("skipped"):
            reason = check.get("reason", "")
            suffix = f" ({reason})" if reason else ""
            print(f"  SKIPPED  {name}{suffix}")
            continue
        if check.get("error"):
            print(f"    ERROR  {name}: {check['error']}")
            continue

        status = "PASS"
        detail = ""

        if name == "PCMCI stability":
            v      = check.get("verdict", "?")
            j      = check.get("mean_jaccard", 0)
            if v == "UNSTABLE":
                status = "WARN"
            detail = f"Jaccard={j:.3f} ({v})"

        elif name == "Significance":
            da    = check.get("test_set_da", 0)
            p     = check.get("binomial_p", 1)
            ci_lo = check.get("bootstrap_ci_low", 0)
            ci_hi = check.get("bootstrap_ci_high", 1)
            if not check.get("binomial_significant", True):
                status = "FAIL"
            detail = f"DA={da:.3f}, p={p:.4f}, 95%CI=[{ci_lo:.3f},{ci_hi:.3f}]"

        elif name == "ARIMA variance":
            if check.get("predict_raw_constant"):
                status = "FAIL"
            std = check.get("predict_raw_std")
            detail = f"std={std:.5f}" if std is not None else ""

        elif name == "Sharpe comparison":
            orig = check.get("original_flat_cost")
            corr = check.get("corrected_turnover")
            if orig is not None and corr is not None:
                detail = f"original={orig:.3f}, corrected={corr:.3f}"

        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        print(f"  {icon}  {status:<6}  {name}"
              + (f"  [{detail}]" if detail else ""))

    print(f"\n  Full results: ml/logs/")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(
        description="Paper-critical validation checks"
    )
    parser.add_argument("--ticker",  type=str, required=True)
    parser.add_argument("--market",  type=str, default="us",
                        choices=["us", "india"])
    parser.add_argument("--quick",   action="store_true",
                        help="Skip slow checks (PCMCI stability, HMM)")
    parser.add_argument("--full",    action="store_true",
                        help="Run ablation table (very slow, ~1hr)")
    args   = parser.parse_args()
    ticker = args.ticker.upper()

    total_start = time.time()
    results = {
        "ticker":   ticker,
        "market":   args.market,
        "run_date": datetime.today().strftime("%Y-%m-%d %H:%M"),
    }

    print(f"\n{'#'*65}")
    print(f"  PAPER VALIDATION — {ticker}  |  Market: {args.market.upper()}")
    print(f"{'#'*65}")

    # Log Ensemble's predict methods to help debug interface mismatches
    try:
        from ml.src.ensemble import Ensemble
        predict_methods = [
            m for m in dir(Ensemble)
            if "predict" in m.lower() and not m.startswith("_")
        ]
        logger.info(f"Ensemble predict methods: {predict_methods}")
    except Exception:
        pass

    try:
        df, df_train, causal_features, target, cfg = load_data(ticker, args.market)
        logger.info(
            f"Loaded: {len(df)} rows, {len(causal_features)} causal features, "
            f"target='{target}'"
        )

        _run_check("check_1", check_1_pcmci_stability,
                   df_train, target, ticker, args.quick, results)

        _run_check("check_2", check_2_nifty_atr,
                   df, args.market, ticker, results)

        _run_check("check_3", check_3_significance,
                   df, causal_features, target, ticker, args.market, results)

        _run_check("check_4", check_4_arima_variance,
                   df, causal_features, target, ticker, results)

        _run_check("check_5", check_5_calibration,
                   df, causal_features, target, ticker, results)

        if args.full:
            banner("CHECK 6: Full Ablation Table")
            try:
                from ml.src.evaluation.ablation import AblationRunner
                from ml.src.causal.granger import GrangerCausality

                granger          = GrangerCausality()
                g_results        = granger.run(df_train, target=target, verbose=False)
                granger_features = granger.get_causal_features(g_results)

                runner      = AblationRunner()
                ablation_df = runner.run(df, ticker, causal_features, granger_features)
                results["ablation_table"] = (
                    ablation_df.to_dict() if not ablation_df.empty else {}
                )
                logger.info("✓ Ablation table complete")
            except ImportError:
                logger.warning("ablation.py not found — copy from corrections/")
                results["ablation_table"] = {"skipped": True}
            except Exception as e:
                logger.error(f"Ablation failed: {e}")
                results["ablation_table"] = {"error": str(e)}

        if not args.quick:
            banner("CHECK 7: HMM Regime Robustness")
            try:
                from ml.src.evaluation.hmm_regime_detector import HMMRegimeDetector
                from ml.src.evaluation.regime_splitter import RegimeSplitter

                detector      = HMMRegimeDetector(n_states=2)
                df_labeled    = detector.fit_label(df)
                splitter      = RegimeSplitter(market=args.market)
                manual_labels = splitter.label(df)["regime"]
                crosstab      = detector.compare_with_manual_regimes(
                    df_labeled, manual_labels
                )
                results["hmm_regime"] = {
                    "states_found": sorted(df_labeled["hmm_regime"].unique().tolist()),
                    "crosstab":     crosstab.to_dict(),
                }
                logger.info("✓ HMM regime detection complete")
            except ImportError as e:
                logger.warning(f"HMM skipped: {e}. pip install hmmlearn")
                results["hmm_regime"] = {"skipped": True, "reason": str(e)}
            except Exception as e:
                logger.error(f"HMM failed: {e}")
                results["hmm_regime"] = {"error": str(e)}

        _run_check("check_8", check_8_sharpe,
                   df, causal_features, target, ticker, results)

    except Exception as e:
        logger.error(f"Fatal validation error: {e}")
        results["fatal_error"] = str(e)
        raise

    finally:
        results["total_elapsed"] = elapsed(total_start)
        log_dir  = ROOT / "ml" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.today().strftime("%Y%m%d_%H%M")
        out_path = log_dir / f"paper_validation_{ticker}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved → {out_path}")

        banner(f"VALIDATION COMPLETE — {elapsed(total_start)}")
        _print_summary(results)


if __name__ == "__main__":
    main()