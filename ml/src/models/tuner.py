"""
tuner.py
--------
Optuna-based hyperparameter tuning for LightGBM and XGBoost.

Finds optimal hyperparameters by minimising validation loss
across a defined number of trials. Best parameters are saved
to config and used for final training.

Usage:
    from ml.src.models.tuner import HyperparameterTuner
    tuner  = HyperparameterTuner()
    params = tuner.tune_lgbm(X_train, y_train, X_val, y_val)
    params = tuner.tune_xgb(X_train, y_train, X_val, y_val)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for ensemble models.
    Optimises directional accuracy on validation set.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.cfg       = _load_config(config_path)
        self.root      = Path(__file__).resolve().parents[4]
        self.n_trials  = self.cfg["features"].get("optuna_trials", 100)
        self.enabled   = self.cfg["features"].get("use_optuna", False)

        self.results_dir = self.root / self.cfg["saved_models"]["dir"] / "tuning"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def tune_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        ticker: str = "default",
    ) -> dict:
        """
        Tune LightGBM hyperparameters with Optuna.
        Returns best params dict ready to pass to LightGBM.
        """
        if not self.enabled:
            logger.info("[tuner] Optuna disabled — using config defaults.")
            return self.cfg["model"]["lgbm"]

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("[tuner] optuna not installed. pip install optuna")
            return self.cfg["model"]["lgbm"]

        import lightgbm as lgb

        logger.info(f"[tuner] Tuning LightGBM ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                "objective":         "regression",
                "metric":            "rmse",
                "verbose":           -1,
                "random_state":      self.cfg["project"]["random_seed"],
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
            model  = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            preds    = model.predict(X_val)
            # Optimise directional accuracy, not just RMSE
            dir_acc  = np.mean(np.sign(preds) == np.sign(y_val.values))
            return -dir_acc   # minimise negative accuracy

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            "objective":    "regression",
            "metric":       "rmse",
            "verbose":      -1,
            "random_state": self.cfg["project"]["random_seed"],
        })

        logger.info(
            f"[tuner] LightGBM best directional accuracy: "
            f"{-study.best_value:.4f} | params: {best_params}"
        )
        self._save_params(ticker, "lgbm", best_params, -study.best_value)
        return best_params

    def tune_xgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        ticker: str = "default",
    ) -> dict:
        """
        Tune XGBoost hyperparameters with Optuna.
        Returns best params dict.
        """
        if not self.enabled:
            return self.cfg["model"]["xgb"]

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("[tuner] optuna not installed.")
            return self.cfg["model"]["xgb"]

        import xgboost as xgb

        logger.info(f"[tuner] Tuning XGBoost ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                "objective":        "reg:squarederror",
                "eval_metric":      "rmse",
                "verbosity":        0,
                "seed":             self.cfg["project"]["random_seed"],
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval   = xgb.DMatrix(X_val,   label=y_val)
            model  = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                callbacks=[xgb.callback.EarlyStopping(50)],
                verbose_eval=False,
            )
            preds   = model.predict(dval)
            dir_acc = np.mean(np.sign(preds) == np.sign(y_val.values))
            return -dir_acc

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            "objective":  "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity":   0,
            "seed":        self.cfg["project"]["random_seed"],
        })

        logger.info(
            f"[tuner] XGBoost best directional accuracy: "
            f"{-study.best_value:.4f}"
        )
        self._save_params(ticker, "xgb", best_params, -study.best_value)
        return best_params

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _save_params(
        self, ticker: str, model: str, params: dict, best_score: float
    ) -> None:
        path = self.results_dir / f"{model}_{ticker}_best_params.json"
        with open(path, "w") as f:
            json.dump({"params": params, "directional_accuracy": best_score}, f, indent=2)
        logger.info(f"[tuner] Best params saved → {path.name}")

    def load_best_params(self, ticker: str, model: str) -> Optional[dict]:
        """Load previously tuned params if they exist."""
        path = self.results_dir / f"{model}_{ticker}_best_params.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        logger.info(
            f"[tuner] Loaded tuned params for {model}/{ticker} "
            f"(DA={data.get('directional_accuracy', 0):.4f})"
        )
        return data["params"]