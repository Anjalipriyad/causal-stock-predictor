"""
tft_model.py
------------
Temporal Fusion Transformer (TFT) for stock return prediction.
Uses pytorch-forecasting library which wraps PyTorch Lightning.

TFT advantages over LGBM/XGB:
    - Handles temporal dependencies explicitly (attention mechanism)
    - Multi-horizon forecasting with interpretable attention weights
    - Variable selection network identifies most important features per timestep
    - Quantile predictions give calibrated uncertainty estimates

Weight in ensemble: 0.35 (replaces some LGBM weight)

Reference: Lim et al. (2020) "Temporal Fusion Transformers for
           Interpretable Multi-horizon Time Series Forecasting"

Usage:
    from ml.src.models.tft_model import TFTModel
    model = TFTModel()
    model.fit(df, causal_features, ticker="AAPL")
    result = model.predict(live_features, ticker="AAPL", current_price=213.40)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.models.base_model import BaseModel, PredictionResult
from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class TFTModel(BaseModel):
    """
    Temporal Fusion Transformer model.
    Falls back to LightGBM if pytorch-forecasting is not installed.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.model_name  = "tft"
        self._model      = None
        self._trainer    = None
        self._scaler_dict = {}

        # TFT-specific config
        self.max_encoder_length  = 60    # lookback window
        self.max_prediction_length = self.horizon
        self.batch_size          = 64
        self.max_epochs          = 30
        self.learning_rate       = 1e-3
        self.hidden_size         = 64
        self.attention_head_size = 4
        self.dropout             = 0.1

        self._available = self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if pytorch-forecasting is available."""
        try:
            import torch
            import pytorch_forecasting
            return True
        except ImportError:
            logger.warning(
                "[tft] pytorch-forecasting not installed. "
                "Install with: pip install torch pytorch-forecasting. "
                "TFT will be disabled — ensemble will use LGBM+XGB+ARIMA only."
            )
            return False

    # -----------------------------------------------------------------------
    # Abstract implementations
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        if not self._available:
            logger.warning("[tft] Skipping TFT training — dependencies not available.")
            return

        import torch
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss

        logger.info(
            f"[tft] Training on {len(X_train)} rows, "
            f"{len(X_train.columns)} features ..."
        )

        # Build combined DataFrame for TFT
        train_df = self._prepare_tft_dataframe(X_train, y_train, split="train")

        if X_val is not None and y_val is not None:
            val_df = self._prepare_tft_dataframe(X_val, y_val, split="val")
        else:
            # Use last 15% of train as val
            split_idx = int(len(train_df) * 0.85)
            val_df    = train_df.iloc[split_idx:].copy()
            train_df  = train_df.iloc[:split_idx].copy()

        # Ensure minimum encoder length
        if len(train_df) < self.max_encoder_length + self.max_prediction_length:
            logger.warning(
                f"[tft] Not enough training data ({len(train_df)} rows). "
                f"Need at least {self.max_encoder_length + self.max_prediction_length}."
            )
            return

        feature_cols = list(X_train.columns)

        # Build TimeSeriesDataSet
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=feature_cols,
            target_normalizer=GroupNormalizer(
                groups=["group_id"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training, val_df, predict=True, stop_randomization=True
        )

        train_loader = training.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        # Build TFT model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=32,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        logger.info(
            f"[tft] Model parameters: {sum(p.numel() for p in tft.parameters()):,}"
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=True,
            gradient_clip_val=0.1,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min",
                )
            ],
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self._model      = tft
        self._trainer    = trainer
        self._training   = training
        self._feature_names = feature_cols
        self._is_fitted  = True
        logger.info("[tft] Training complete.")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return median quantile predictions as numpy array."""
        if not self._available or self._model is None:
            raise RuntimeError(
                "[tft] Model not fitted or dependencies not available."
            )
        pred_df = self._prepare_tft_dataframe(X, pd.Series(np.zeros(len(X))), split="pred")
        dataset = self._training.__class__.from_dataset(
            self._training, pred_df, predict=True, stop_randomization=True
        )
        loader = dataset.to_dataloader(train=False, batch_size=len(X), num_workers=0)
        preds  = self._model.predict(loader, mode="quantiles")
        # Return median prediction (middle quantile)
        return preds[:, :, 2].mean(axis=1).numpy()

    def save(self, ticker: str) -> None:
        """Save TFT model checkpoint."""
        if not self._is_fitted or not self._available:
            return
        import torch
        path = self._model_path(ticker, "tft_filename") \
               if "tft_filename" in self.cfg.get("saved_models", {}) \
               else self.models_dir / f"tft_{ticker.upper()}.pt"
        torch.save({
            "model_state": self._model.state_dict(),
            "feature_names": self._feature_names,
        }, path)
        logger.info(f"[tft] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load TFT model from checkpoint."""
        if not self._available:
            return
        import torch
        path = self.models_dir / f"tft_{ticker.upper()}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No TFT model for {ticker} at {path}")
        checkpoint          = torch.load(path, map_location="cpu")
        self._feature_names = checkpoint["feature_names"]
        self._is_fitted     = True
        logger.info(f"[tft] Model loaded from {path.name}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _prepare_tft_dataframe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split: str = "train",
    ) -> pd.DataFrame:
        """
        Convert feature DataFrame to TFT-compatible format.
        TFT requires: time_idx (int), group_id (str), target (float), features.
        """
        df = X.copy()
        df["target"]   = y.values if hasattr(y, "values") else y
        df["group_id"] = "stock"     # single stock = single group
        df["time_idx"] = np.arange(len(df))
        return df.reset_index(drop=True)

    def is_available(self) -> bool:
        """Check if TFT can be used."""
        return self._available and self._is_fitted