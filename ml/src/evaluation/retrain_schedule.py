"""
retrain_schedule.py
-------------------
Handles periodic retraining of models on fresh data.
Called by run_pipeline.py when --retrain flag is set.

Strategy: expanding window retraining every 6 months.
Each retrain uses ALL data up to that point (not a fixed window).
This keeps the model current without discarding historical signal.

Usage:
    from ml.src.evaluation.retrain_schedule import RetrainScheduler
    scheduler = RetrainScheduler()
    scheduler.run(df, ticker="AAPL", causal_features=features)
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """
    Manages periodic retraining on expanding data windows.
    Saves a new model checkpoint for each retraining window.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg    = _load_config(config_path)
        self.root   = Path(__file__).resolve().parents[3]
        self.target = self.cfg["model"]["target"]

        bt = self.cfg["evaluation"]["backtest"]
        self.initial_train_years = bt["initial_train_years"]
        self.step_size_months    = bt["step_size_months"]
        self.min_test_samples    = bt["min_test_samples"]

    def should_retrain(self, ticker: str) -> bool:
        """
        Check if models need retraining based on last saved timestamp.
        Returns True if no models exist or last train > 6 months ago.
        """
        models_dir = self.root / self.cfg["saved_models"]["dir"]
        lgbm_path  = models_dir / f"lgbm_{ticker.upper()}.pkl"

        if not lgbm_path.exists():
            return True

        # Check modification time
        mtime     = lgbm_path.stat().st_mtime
        last_train = datetime.fromtimestamp(mtime)
        months_ago = (datetime.now() - last_train).days / 30

        if months_ago > 6:
            logger.info(
                f"[retrain] Models for {ticker} are {months_ago:.1f} months old — retraining."
            )
            return True

        logger.info(
            f"[retrain] Models for {ticker} are {months_ago:.1f} months old — still fresh."
        )
        return False

    def run(
        self,
        df: pd.DataFrame,
        ticker: str,
        causal_features: list[str],
        force: bool = False,
    ) -> dict:
        """
        Run walk-forward retraining evaluation.

        For each 6-month window from initial_train_years onward:
            1. Train on all data up to window end
            2. Evaluate on next 6 months
            3. Save best model checkpoint

        Args:
            df:              Full feature matrix
            ticker:          e.g. "AAPL"
            causal_features: Causal feature list
            force:           Retrain even if models are fresh

        Returns:
            Dict of {window_label: metrics} for each training window
        """
        from ml.src.ensemble import Ensemble
        from ml.src.evaluation.metrics import Metrics

        if not force and not self.should_retrain(ticker):
            logger.info(f"[retrain] Skipping retraining for {ticker}.")
            return {}

        ticker  = ticker.upper()
        metrics = Metrics()
        results = {}

        # Find initial train end
        start_dt  = df.index[0]
        init_end  = start_dt + pd.DateOffset(years=self.initial_train_years)
        step      = pd.DateOffset(months=self.step_size_months)

        window_end = df.index[df.index <= init_end][-1]
        best_da    = 0.0
        best_window = None

        logger.info(
            f"[retrain] Starting walk-forward retraining for {ticker} "
            f"from {window_end.date()} ..."
        )

        while window_end < df.index[-1]:
            test_start = window_end + pd.Timedelta(days=1)
            test_end   = min(window_end + step, df.index[-1])

            train_df = df.loc[:window_end]
            test_df  = df.loc[test_start:test_end]

            if len(test_df) < self.min_test_samples:
                window_end = test_end
                continue

            label = f"{window_end.strftime('%Y%m')}"
            logger.info(
                f"[retrain] Window {label}: "
                f"train={len(train_df)}, test={len(test_df)}"
            )

            try:
                ensemble = Ensemble()
                ensemble.train_all(train_df, ticker, causal_features)

                # CRITICAL (Issue #2): Extract the implicit inner validation set from train_df
                # to track checkpoint fitness, preventing walk-forward test_df leakage.
                n_train = len(train_df)
                val_start_idx = int(n_train * 0.70)
                test_start_idx = int(n_train * 0.85)
                inner_val_df = train_df.iloc[val_start_idx:test_start_idx]
                
                val_preds = ensemble.predict_historical(inner_val_df, causal_features)
                if "actual_return" in val_preds.columns:
                    val_scores = metrics.compute_all(
                        val_preds["predicted_return"],
                        val_preds["actual_return"],
                        label=f"{label}_val",
                    )
                    val_da = val_scores.get("directional_accuracy", 0)
                    if val_da > best_da:
                        best_da     = val_da
                        best_window = label
                        self._save_best(ticker, ensemble, label)

                # CRITICAL (Issue #5): Evaluate actual walk-forward test holdout for reporting
                preds = ensemble.predict_historical(test_df, causal_features)

                if "actual_return" not in preds.columns:
                    window_end = test_end
                    continue

                scores = metrics.compute_all(
                    preds["predicted_return"],
                    preds["actual_return"],
                    label=label,
                )
                results[label] = scores

            except Exception as e:
                logger.warning(f"[retrain] Window {label} failed: {e}")

            window_end = test_end

        if best_window:
            logger.info(
                f"[retrain] Best window: {best_window} "
                f"(DA={best_da:.4f}) — promoted to production model."
            )
            self._promote_best(ticker, best_window)

        return results

    def _save_best(self, ticker: str, ensemble, window_label: str) -> None:
        """Save model checkpoint for a specific window."""
        import joblib
        models_dir = self.root / self.cfg["saved_models"]["dir"]
        checkpoint_dir = models_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        for model_name, model_obj in [
            ("lgbm", ensemble.lgbm),
            ("xgb",  ensemble.xgb),
            ("arima", ensemble.arima),
        ]:
            path = checkpoint_dir / f"{model_name}_{ticker}_{window_label}.pkl"
            try:
                joblib.dump(model_obj, path)
            except Exception as e:
                logger.warning(f"[retrain] Could not save checkpoint {path}: {e}")

    def _promote_best(self, ticker: str, window_label: str) -> None:
        """Copy best checkpoint to production model path."""
        import shutil
        models_dir     = self.root / self.cfg["saved_models"]["dir"]
        checkpoint_dir = models_dir / "checkpoints"

        for model_name in ["lgbm", "xgb", "arima"]:
            src  = checkpoint_dir / f"{model_name}_{ticker}_{window_label}.pkl"
            dest = models_dir / f"{model_name}_{ticker}.pkl"
            if src.exists():
                shutil.copy2(src, dest)
                logger.info(f"[retrain] Promoted {src.name} → {dest.name}")