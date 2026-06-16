"""
leakage_guard.py
----------------
Shared helpers for keeping forward-looking return labels out of feature sets.

The active target column is allowed as the supervised label for Granger/PCMCI
on the training slice. Every other forward-return label must be removed before
feature selection, ablations, baselines, and model inputs.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


NON_FEATURE_LABELS = {"direction", "target_q"}


def is_forward_return_label(col: str) -> bool:
    """Return True for columns that encode future returns or derived labels."""
    if col in NON_FEATURE_LABELS:
        return True
    if col.startswith("excess_return_"):
        return True
    if col.startswith("log_return_") and col != "log_return_1d":
        return True
    if "_sm_" in col and (
        col.startswith("excess_return_") or col.startswith("log_return_")
    ):
        return True
    return False


def forward_return_columns(
    columns: Iterable[str],
    *,
    active_target: str | None = None,
    keep_active_target: bool = False,
) -> list[str]:
    """
    List forward-looking columns that should be dropped from a feature set.

    Args:
        columns: Candidate column names.
        active_target: The supervised label for this run.
        keep_active_target: Keep active_target when building a causal-discovery
            frame that needs the label column present.
    """
    to_drop: list[str] = []
    for col in columns:
        if keep_active_target and active_target is not None and col == active_target:
            continue
        if is_forward_return_label(col):
            to_drop.append(col)
    return to_drop


def safe_feature_columns(
    df: pd.DataFrame,
    *,
    active_target: str | None = None,
) -> list[str]:
    """Return feature columns with all future-return labels removed."""
    drops = set(
        forward_return_columns(
            df.columns,
            active_target=active_target,
            keep_active_target=False,
        )
    )
    if active_target is not None:
        drops.add(active_target)
    return [c for c in df.columns if c not in drops]


def strip_forward_return_labels(
    df: pd.DataFrame,
    *,
    active_target: str | None = None,
    keep_active_target: bool = False,
) -> pd.DataFrame:
    """Drop forward-looking labels, optionally retaining the active target."""
    drops = forward_return_columns(
        df.columns,
        active_target=active_target,
        keep_active_target=keep_active_target,
    )
    return df.drop(columns=drops, errors="ignore")


def make_causal_discovery_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build a frame for supervised causal discovery.

    All auxiliary future-return labels are stripped. The active target is then
    re-attached as the only label column, allowing Granger/PCMCI to test
    feature-to-target links without exposing other answer columns as features.
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")
    clean = strip_forward_return_labels(
        df,
        active_target=target,
        keep_active_target=False,
    ).copy()
    clean[target] = df[target]
    return clean
