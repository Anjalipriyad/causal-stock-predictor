"""
lstm_model.py
-------------
LSTM base learner for the stacking ensemble.

Role in the stack:
    LSTM captures temporal patterns that tree models cannot.
    LightGBM/XGBoost see one flat feature row per prediction.
    LSTM sees the last `sequence_length` rows as an ordered sequence,
    allowing it to learn patterns like:
        "3 consecutive days of rising VIX → next-day selloff"
        "RSI overbought + declining momentum → reversal incoming"

Weight in ensemble: 0.25 (when LSTM available; LightGBM reduces to 0.40,
XGBoost to 0.25, ARIMA stays 0.10). Configured via stacking meta-learner.

Architecture:
    2-layer LSTM with dropout + fully connected output
    Input: (batch, sequence_length=20, n_features)
    Output: (batch,) scalar log return prediction

Key design decisions:
    - sequence_length=20 trading days (~1 calendar month of lookback)
    - Gradient clipping at 1.0 prevents exploding gradients on financial data
    - Early stopping with patience=10 on validation loss
    - Falls back gracefully to zeros if PyTorch not installed — the stacking
      meta-learner will assign zero weight to a model that predicts all zeros,
      effectively removing it from the ensemble without crashing

Usage:
    from ml.src.models.lstm_model import LSTMModel
    model = LSTMModel()
    if model.is_available():
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        preds = model.predict_raw(X_test_scaled)
        model.save("AAPL")
        model.load("AAPL")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.models.base_model import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM sequence model for 5-day return prediction.
    Slots into the stacking ensemble as a 4th base learner.
    Falls back gracefully if PyTorch is not installed.
    """

    def __init__(self, config_path: Optional[str] = None, cfg: Optional[dict] = None):
        super().__init__(config_path, cfg)
        self.model_name      = "lstm"
        self._model          = None
        self._feature_names: list[str] = []

        # Sequence length: how many past trading days the LSTM sees
        # 20 days = ~1 calendar month, balances memory vs noise
        self.sequence_length = 20
        self.hidden_size     = 64    # LSTM hidden units per layer
        self.num_layers      = 2     # stacked LSTM layers
        self.dropout         = 0.2   # between LSTM layers
        self.batch_size      = 32
        self.max_epochs      = 50
        self.learning_rate   = 1e-3
        self.patience        = 10    # early stopping patience

        self._available = self._check_torch()
        self._best_state: Optional[dict] = None

    def is_available(self) -> bool:
        """Returns True if PyTorch is installed and LSTM can be used."""
        return self._available

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
        """
        Train LSTM on the sequence data.

        X_train must have at least sequence_length+1 rows.
        If PyTorch is not available, logs a warning and returns without training.
        """
        if not self._available:
            logger.warning(
                "[lstm] PyTorch not installed — LSTM training skipped. "
                "Install with: pip install torch. "
                "Ensemble will fall back to LightGBM+XGBoost+ARIMA only."
            )
            return

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        self._feature_names = list(X_train.columns)
        n_features          = len(self._feature_names)

        # Minimum rows check
        min_rows = self.sequence_length + 1
        if len(X_train) < min_rows:
            logger.warning(
                f"[lstm] X_train has {len(X_train)} rows but needs {min_rows} "
                f"for sequence_length={self.sequence_length}. Skipping training."
            )
            return

        logger.info(
            f"[lstm] Training on {len(X_train)} rows, "
            f"{n_features} features, "
            f"sequence_length={self.sequence_length} ..."
        )

        X_t, y_t = self._build_sequences(X_train, y_train)

        # ── Build model ────────────────────────────────────────────────────
        model = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        loader    = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=False,   # DO NOT shuffle — time series order matters
        )

        # ── Training loop with early stopping ─────────────────────────────
        best_val_loss     = float("inf")
        patience_counter  = 0
        self._best_state  = None

        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                # Gradient clipping prevents exploding gradients on volatile financial data
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation loss for early stopping
            if X_val is not None and y_val is not None and len(X_val) > min_rows:
                model.eval()
                with torch.no_grad():
                    X_v, y_v = self._build_sequences(X_val, y_val)
                    if len(X_v) > 0:
                        val_loss = criterion(model(X_v), y_v).item()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            # Deep copy state dict — can't use reference
                            self._best_state = {
                                k: v.clone() for k, v in model.state_dict().items()
                            }
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= self.patience:
                            logger.info(
                                f"[lstm] Early stopping at epoch {epoch + 1} "
                                f"(val_loss={best_val_loss:.6f})"
                            )
                            break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[lstm] Epoch {epoch + 1}/{self.max_epochs} — "
                    f"train_loss={train_loss / len(loader):.6f}"
                )

        # Restore best weights
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

        self._model     = model
        self._is_fitted = True
        logger.info("[lstm] Training complete.")

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return scalar predictions for each row of X.

        Important: LSTM needs sequence_length rows to make a prediction.
        For the first sequence_length rows, we return the first valid prediction
        (i.e. pad the front). This ensures the output array is always len(X),
        matching what LightGBM and XGBoost return.
        """
        if not self._available:
            raise RuntimeError(
                "[lstm] PyTorch not installed. "
                "Cannot call predict_raw() on unfitted LSTM."
            )
        if self._model is None or not self._is_fitted:
            raise RuntimeError(
                "[lstm] Model not fitted. Call fit() first."
            )

        import torch

        self._model.eval()

        if len(X) < self.sequence_length + 1:
            # Not enough rows for even one sequence — return zeros
            logger.warning(
                f"[lstm] X has only {len(X)} rows "
                f"(need {self.sequence_length + 1}). Returning zeros."
            )
            return np.zeros(len(X))

        X_seq = self._build_sequences_inference(X)

        with torch.no_grad():
            preds_seq = self._model(X_seq).numpy()   # shape: (n_valid,)

        # Pad the first sequence_length rows with the first prediction
        # so the output length matches len(X)
        padding = np.full(self.sequence_length, preds_seq[0])
        return np.concatenate([padding, preds_seq])

    def save(self, ticker: str) -> None:
        """Save LSTM model weights to saved_models/."""
        if not self._available or not self._is_fitted or self._model is None:
            logger.warning("[lstm] Not fitted or PyTorch unavailable — nothing to save.")
            return
        import torch
        path = self.models_dir / f"lstm_{ticker.upper()}.pt"
        torch.save({
            "state_dict":     self._model.state_dict(),
            "feature_names":  self._feature_names,
            "n_features":     len(self._feature_names),
            "hidden_size":    self.hidden_size,
            "num_layers":     self.num_layers,
            "dropout":        self.dropout,
            "sequence_length": self.sequence_length,
        }, path)
        logger.info(f"[lstm] Model saved → {path.name}")

    def load(self, ticker: str) -> None:
        """Load LSTM model weights from saved_models/."""
        if not self._available:
            logger.warning("[lstm] PyTorch not available — cannot load LSTM model.")
            return
        import torch
        path = self.models_dir / f"lstm_{ticker.upper()}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"No LSTM model found for {ticker} at {path}. "
                "Run fit() + save() first."
            )
        checkpoint = torch.load(path, map_location="cpu")
        n_features = checkpoint["n_features"]

        self._model = _LSTMNet(
            n_features  = n_features,
            hidden_size = checkpoint["hidden_size"],
            num_layers  = checkpoint["num_layers"],
            dropout     = checkpoint["dropout"],
        )
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
        self._feature_names  = checkpoint["feature_names"]
        self.sequence_length = checkpoint["sequence_length"]
        self._is_fitted      = True
        logger.info(
            f"[lstm] Model loaded from {path.name} "
            f"(sequence_length={self.sequence_length}, "
            f"hidden={checkpoint['hidden_size']})"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _check_torch(self) -> bool:
        """Check if PyTorch is installed."""
        try:
            import torch
            return True
        except ImportError:
            logger.warning(
                "[lstm] PyTorch not installed — LSTM disabled. "
                "To enable: pip install torch"
            )
            return False

    def _build_sequences(
        self, X: pd.DataFrame, y: pd.Series
    ):
        """
        Convert flat feature matrix into overlapping sequences.

        Input:  X of shape (n, features), y of shape (n,)
        Output: X_tensor of shape (n - seq_len, seq_len, features)
                y_tensor of shape (n - seq_len,)

        Row i of X_tensor contains rows [i : i+seq_len] of X.
        Row i of y_tensor contains y[i+seq_len] (the target for that sequence).
        """
        import torch
        data    = X.values.astype(np.float32)
        targets = y.values.astype(np.float32)
        seq     = self.sequence_length
        n       = len(data)

        X_seqs = np.array([data[i : i + seq] for i in range(n - seq)])
        y_seqs = targets[seq:]

        return torch.tensor(X_seqs), torch.tensor(y_seqs)

    def _build_sequences_inference(self, X: pd.DataFrame):
        """
        Build sequences for inference (no targets needed).
        Returns tensor of shape (n - seq_len, seq_len, features).
        """
        import torch
        data = X.values.astype(np.float32)
        seq  = self.sequence_length
        n    = len(data)
        X_seqs = np.array([data[i : i + seq] for i in range(n - seq)])
        return torch.tensor(X_seqs)


# ── PyTorch module (defined at module level for pickle compatibility) ─────────

class _LSTMNet:
    """
    2-layer stacked LSTM with fully connected output head.
    Defined as a standalone class (not nested) so joblib/pickle works correctly
    when saving and loading.
    """

    def __new__(cls, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        """Dynamically create the nn.Module only when PyTorch is available."""
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, n_feat, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_feat,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    dropout=drop if layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(drop)
                self.fc      = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: (batch, seq_len, features)
                out, _ = self.lstm(x)
                # Use only the last timestep's hidden state
                out = self.dropout(out[:, -1, :])
                return self.fc(out).squeeze(-1)   # (batch,)

        return _Net(n_features, hidden_size, num_layers, dropout)