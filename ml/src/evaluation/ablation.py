"""
ablation.py
-----------
Ablation study runner for the paper's Table 3 (ablation table).

The original backtester.regime_backtest() only compares:
    - pcmci_causal
    - all_features
    - random

This is insufficient for the paper. Reviewers will ask:
    "Is PCMCI better than Granger alone?"
    "Is the ensemble better than just LightGBM?"
    "Is causal selection adding value over just using fewer features?"

This module adds:
    1. granger_only:   LightGBM trained on Granger-selected features
    2. arima_only:     ARIMA model alone (no LightGBM/XGB)
    3. lgbm_only:      LightGBM alone (no blending) on causal features
    4. top_k_features: LightGBM on top-k features by LGBM importance
                       (sanity check: is PCMCI adding value over
                       just picking the features the model already likes?)

Paper Table 3 structure:
    Model               | bull | covid_crash | recovery | rate_hike | ai_bull
    pcmci_causal        |  DA  |     DA      |    DA    |    DA     |   DA
    granger_only        |  DA  |     DA      |    DA    |    DA     |   DA
    all_features        |  DA  |     DA      |    DA    |    DA     |   DA
    lgbm_only (causal)  |  DA  |     DA      |    DA    |    DA     |   DA
    arima_only          |  DA  |     DA      |    DA    |    DA     |   DA
    top_k_importance    |  DA  |     DA      |    DA    |    DA     |   DA
    random              |  DA  |     DA      |    DA    |    DA     |   DA

Usage:
    from ml.src.evaluation.ablation import AblationRunner
    runner = AblationRunner()
    results = runner.run(df, ticker="AAPL", causal_features=features,
                         granger_features=granger_features)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config
from ml.src.evaluation.metrics import Metrics
from ml.src.evaluation.regime_splitter import RegimeSplitter
from ml.src.causal.leakage_guard import make_causal_discovery_frame, safe_feature_columns

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs the full ablation study comparing all model variants.
    Produces the table needed to establish PCMCI's contribution.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg     = _load_config(config_path)
        self.metrics = Metrics(config_path)
        self.splitter = RegimeSplitter(config_path)
        self.target   = self.cfg["model"]["target"]

        bt = self.cfg["evaluation"]["backtest"]
        self.min_test_samples = bt["min_test_samples"]

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        granger_features: list[str],
        top_k: int = 10,
        config_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run all ablation variants across all regimes.
        """
        logger.info(f"[ablation] Running full ablation study for {ticker} ...")

        # 1. Determine target column: check if excess_return_5d has real signal (not all zeros), otherwise use log_return_5d
        target_col = "log_return_5d"
        if "excess_return_5d" in df.columns:
            if df["excess_return_5d"].abs().sum() > 0:
                target_col = "excess_return_5d"
        self.target = target_col
        logger.info(f"[ablation] Using target column: {self.target}")

        # 2. Prepare features
        all_feature_cols = safe_feature_columns(df, active_target=self.target)
        safe_features = set(all_feature_cols)

        causal_features = [f for f in causal_features if f in safe_features]
        granger_features = [f for f in granger_features if f in safe_features]
        
        # Compute intersection
        intersection_features = list(set(causal_features).intersection(set(granger_features)))
        logger.info(f"[ablation] PCMCI features: {len(causal_features)}, Granger features: {len(granger_features)}, Intersection: {len(intersection_features)}")

        # Variants definitions
        variants = {
            "all_features": all_feature_cols,
            "granger_only": granger_features,
            "pcmci_only": causal_features,
            "intersection": intersection_features,
        }

        regime_splits = self.splitter.split_all(df)
        
        # Store results and predictions
        results_list = []
        predictions_by_model_regime = {}
        y_true_by_regime = {}

        # 1. Evaluate Model Variants
        for model_name, features in variants.items():
            if not features:
                logger.warning(f"[ablation] {model_name}: empty feature list, using fallback.")
                features = all_feature_cols[:2] if not features else features
                
            logger.info(f"[ablation] Testing model variant: {model_name} ({len(features)} features)")

            for regime_name, regime_df in regime_splits.items():
                if len(regime_df) < self.min_test_samples:
                    continue

                regime_start = self.splitter.regimes[regime_name][0]
                train_df = df.loc[:regime_start].iloc[:-1]
                if len(train_df) < 200:
                    continue

                res = self._run_lgbm_only(
                    train_df, regime_df, ticker, features, regime_name, config_path
                )
                if res:
                    scores, y_pred, y_true = res
                    scores["model"] = model_name
                    scores["regime"] = regime_name
                    scores["n_test"] = len(y_true)
                    results_list.append(scores)
                    predictions_by_model_regime[(model_name, regime_name)] = y_pred
                    y_true_by_regime[regime_name] = y_true

        # 2. Evaluate Baselines
        for regime_name, regime_df in regime_splits.items():
            if len(regime_df) < self.min_test_samples:
                continue
            regime_start = self.splitter.regimes[regime_name][0]
            train_df = df.loc[:regime_start].iloc[:-1]
            if len(train_df) < 200:
                continue

            y_true = regime_df[self.target].dropna()
            if len(y_true) < self.min_test_samples:
                continue

            # persistence (predict UP if prior 5-day return was positive, DOWN if negative)
            persistence_raw = df[self.target].shift(5).loc[y_true.index].fillna(0.0)
            y_pred_persist = np.where(persistence_raw >= 0, 1.0, -1.0)
            y_pred_persist = pd.Series(y_pred_persist, index=y_true.index)

            scores_persist = self.metrics.compute_all(y_pred_persist, y_true, label=f"persistence/{regime_name}")
            scores_persist["model"] = "persistence"
            scores_persist["regime"] = regime_name
            scores_persist["n_test"] = len(y_true)
            results_list.append(scores_persist)
            predictions_by_model_regime[("persistence", regime_name)] = y_pred_persist

            # buy_and_hold (always predict UP)
            y_pred_bh = pd.Series(1.0, index=y_true.index)
            scores_bh = self.metrics.compute_all(y_pred_bh, y_true, label=f"buy_and_hold/{regime_name}")
            scores_bh["model"] = "buy_and_hold"
            scores_bh["regime"] = regime_name
            scores_bh["n_test"] = len(y_true)
            results_list.append(scores_bh)
            predictions_by_model_regime[("buy_and_hold", regime_name)] = y_pred_bh
            y_true_by_regime[regime_name] = y_true

        if not results_list:
            logger.error("[ablation] No results computed.")
            return pd.DataFrame()

        result_df = pd.DataFrame(results_list)
        
        # 3. Post-Process Metrics to Add 95% CI and Effective N
        from ml.src.evaluation.significance import SignificanceTester
        tester = SignificanceTester()
        
        ci_lows = []
        ci_highs = []
        effective_ns = []
        mcnemar_pvals = []
        
        for idx, row in result_df.iterrows():
            model = row["model"]
            regime = row["regime"]
            n_test = int(row["n_test"])
            
            eff_n = n_test // 5
            effective_ns.append(eff_n)
            
            y_pred = predictions_by_model_regime.get((model, regime))
            y_true = y_true_by_regime.get(regime)
            
            if y_pred is not None and y_true is not None:
                # Compute Block Bootstrap CI
                boot_res = tester.bootstrap_da_ci(y_pred, y_true, horizon_days=5)
                ci_lows.append(boot_res.ci_low)
                ci_highs.append(boot_res.ci_high)
                
                # Compute McNemar's p-value (model vs all_features)
                y_pred_all = predictions_by_model_regime.get(("all_features", regime))
                if y_pred_all is not None:
                    mcnemar_res = tester.mcnemar_test(y_pred, y_pred_all, y_true, label_a=model, label_b="all_features")
                    mcnemar_pvals.append(mcnemar_res.p_value)
                else:
                    mcnemar_pvals.append(np.nan)
            else:
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
                mcnemar_pvals.append(np.nan)
                
        result_df["ci_low"] = ci_lows
        result_df["ci_high"] = ci_highs
        result_df["n_effective"] = effective_ns
        result_df["mcnemar_pval"] = mcnemar_pvals
        
        # Set Multi-index
        result_df = result_df.set_index(["model", "regime"]).sort_index()
        
        # Save to CSV
        root_path = Path(__file__).resolve().parents[3]
        out_dir = root_path / "paper_output" / "NIFTY"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "table2_ablation.csv"
        
        result_df.to_csv(out_path)
        logger.info(f"[ablation] Results saved to {out_path}")
        
        # Print formatted table
        self._print_ablation_table(result_df)
        
        return result_df

    def _run_lgbm_only(
        self,
        train_df:    pd.DataFrame,
        test_df:     pd.DataFrame,
        ticker:      str,
        features:    list[str],
        regime_name: str,
        config_path: Optional[str],
    ) -> Optional[tuple[dict, pd.Series, pd.Series]]:
        """Train LightGBM alone (no blending) and evaluate."""
        from ml.src.models.lgbm_model import LGBMModel
        try:
            model = LGBMModel(config_path)
            X_tr, X_va, _, y_tr, y_va, _ = model.prepare_data(train_df, features)
            X_tr_s, X_va_s, _ = model.scale(X_tr, X_va, X_tr, ticker)   # scale
            model.fit(X_tr_s, y_tr, X_va_s, y_va)

            feat_cols  = [c for c in features if c in test_df.columns]
            X_test     = test_df[feat_cols]
            X_test_s   = model.transform(X_test)
            y_pred     = pd.Series(model.predict_raw(X_test_s), index=X_test.index)
            y_true     = test_df[self.target] if self.target in test_df.columns else None

            if y_true is None:
                return None

            scores = self.metrics.compute_all(y_pred, y_true, label=f"lgbm_only/{regime_name}")
            return scores, y_pred, y_true
        except Exception as e:
            logger.warning(f"[ablation] lgbm_only/{regime_name}: {e}")
            return None

    def _print_ablation_table(self, result_df: pd.DataFrame) -> None:
        """Print highly polished table for the paper's central claims."""
        print("\n" + "="*120)
        print("CENTRAL ABLATION STUDY: DOES PCMCI FEATURE SELECTION ADD VALUE?")
        print("="*120)
        
        regimes = result_df.index.get_level_values("regime").unique()
        models = ["buy_and_hold", "persistence", "all_features", "granger_only", "intersection", "pcmci_only"]
        
        for regime in regimes:
            first_row = result_df.xs(regime, level="regime").iloc[0]
            n_eff = int(first_row.get("n_effective", 0))
            unreliable_str = " [UNRELIABLE]" if n_eff < 20 else ""
            
            print(f"\nREGIME: {regime.upper()}{unreliable_str} (N_test={int(first_row['n_test'])}, N_effective={n_eff})")
            print("-" * 120)
            print(f"{'Model Variant':<20} | {'DA':<6} | {'95% Block Bootstrap CI':<24} | {'Sharpe':<8} | {'RMSE':<8} | {'McNemar p (vs All)':<20}")
            print("-" * 120)
            
            for model in models:
                if (model, regime) not in result_df.index:
                    continue
                row = result_df.loc[(model, regime)]
                
                da = row["directional_accuracy"]
                ci_low = row["ci_low"]
                ci_high = row["ci_high"]
                sharpe = row["sharpe_ratio"]
                rmse = row["rmse"]
                
                ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]" if not pd.isna(ci_low) else "N/A"
                
                p_val = row.get("mcnemar_pval", np.nan)
                if model == "pcmci_only":
                    p_str = f"{p_val:.4f}*" if p_val < 0.05 else f"{p_val:.4f}"
                elif model == "all_features":
                    p_str = "Baseline"
                else:
                    p_str = "-"
                    
                print(f"{model:<20} | {da:.4f} | {ci_str:<24} | {sharpe:.4f} | {rmse:.5f} | {p_str:<20}")
            print("-" * 120)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from ml.src.causal.granger import GrangerCausality
    from ml.src.causal.selector import CausalSelector

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--ticker",  type=str, required=True)
    parser.add_argument("--market",  type=str, default="us",
                        choices=["us", "india"])
    parser.add_argument("--top-k",   type=int, default=10)
    args   = parser.parse_args()

    ticker = args.ticker.upper()
    cfg    = _load_config()

    if args.market == "india" or ticker in ("NIFTY", "^NSEI"):
        from ml.src.data.nifty_loader import NiftyLoader
        feat_path = NiftyLoader().out_dir / "NIFTY_features.csv"
        target    = "log_return_5d"
    else:
        from ml.src.features.pipeline import FeaturePipeline
        feat_path = FeaturePipeline().features_dir / f"{ticker}_features.csv"
        target    = cfg["model"]["target"]

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Load causal features
    causal_features  = CausalSelector().load(ticker)

    # Rerun Granger to get Granger-only features
    train_end        = int(len(df) * cfg["model"]["train_ratio"])
    df_train         = make_causal_discovery_frame(df.iloc[:train_end], target)
    granger          = GrangerCausality()
    granger_results  = granger.run(df_train, target=target, verbose=False)
    granger_features = granger.get_causal_features(granger_results)

    runner  = AblationRunner()
    results = runner.run(df, ticker, causal_features, granger_features, top_k=args.top_k)
    print(results)
