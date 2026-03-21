"""
finbert.py
----------
FinBERT-based sentiment scoring — replaces keyword-based sentiment.

FinBERT is a BERT model fine-tuned on financial text. It understands
context, negation, and financial jargon that keyword models miss.

Keyword model: "not good quarter"  → positive (sees "good")
FinBERT:       "not good quarter"  → negative (understands context)

Model: ProsusAI/finbert (HuggingFace)
Outputs: positive, negative, neutral probabilities per headline
We use: positive_prob - negative_prob as the sentiment score

Note: Slow on CPU (~1-2 sec per headline). Set finbert.enabled: false
      in config if you don't have GPU. Keyword fallback is automatic.

Usage:
    from ml.src.features.finbert import FinBERTSentiment
    finbert = FinBERTSentiment()
    score = finbert.score_headline("Apple beats earnings expectations")
    df    = finbert.score_dataframe(news_df)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.src.data.loader import _load_config

logger = logging.getLogger(__name__)


class FinBERTSentiment:
    """
    FinBERT-based financial sentiment scorer.
    Falls back to keyword scoring if transformers not installed or
    finbert.enabled is false in config.
    """

    # Keyword fallback lexicons
    POSITIVE_WORDS = {
        "beat", "beats", "surge", "surges", "rises", "gain", "gains",
        "profit", "record", "growth", "upgrade", "strong", "bullish",
        "rally", "outperform", "exceed", "exceeds", "positive", "up",
        "boost", "increase", "revenue", "buyback", "dividend", "expansion",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "fall", "falls", "drop", "drops", "loss",
        "losses", "decline", "declines", "downgrade", "weak", "bearish",
        "crash", "slump", "underperform", "below", "negative", "down",
        "risk", "risks", "cut", "cuts", "lawsuit", "investigation", "recall",
    }

    def __init__(self, config_path: Optional[str] = None):
        cfg = _load_config(config_path)
        finbert_cfg      = cfg["features"].get("finbert", {})
        self.enabled     = finbert_cfg.get("enabled", False)
        self.model_name  = finbert_cfg.get("model", "ProsusAI/finbert")
        self.batch_size  = finbert_cfg.get("batch_size", 32)
        self.max_length  = finbert_cfg.get("max_length", 512)
        self.fallback    = finbert_cfg.get("fallback_to_keyword", True)

        self._pipeline   = None
        self._loaded     = False

        if self.enabled:
            self._load_model()

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def score_headline(self, headline: str) -> float:
        """
        Score a single headline.
        Returns float in [-1, 1]:
            +1 = strongly positive
            -1 = strongly negative
             0 = neutral
        """
        if not headline or not isinstance(headline, str):
            return 0.0

        if self._loaded and self._pipeline is not None:
            return self._finbert_score(headline)
        else:
            return self._keyword_score(headline)

    def score_batch(self, headlines: list[str]) -> list[float]:
        """
        Score a batch of headlines efficiently.
        Uses batched inference for FinBERT, keyword scoring as fallback.
        """
        if not headlines:
            return []

        if self._loaded and self._pipeline is not None:
            return self._finbert_batch(headlines)
        else:
            return [self._keyword_score(h) for h in headlines]

    def score_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of news items.
        Expects columns: date, headline (or title)
        Returns DataFrame with date index and sentiment_score column.
        """
        if news_df is None or news_df.empty:
            return pd.DataFrame()

        headline_col = "headline" if "headline" in news_df.columns else "title"
        if headline_col not in news_df.columns:
            logger.warning("[finbert] No headline column found.")
            return pd.DataFrame()

        headlines = news_df[headline_col].fillna("").tolist()
        scores    = self.score_batch(headlines)

        result = news_df.copy()
        result["finbert_score"] = scores

        # Aggregate by date
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])
            daily = result.groupby("date")["finbert_score"].agg(["mean","std","count"])
            daily.columns = ["sentiment_score","sentiment_std","article_count"]
            daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
            return daily

        return result[["finbert_score"]]

    def is_available(self) -> bool:
        """Check if FinBERT model is loaded and ready."""
        return self._loaded and self._pipeline is not None

    # -----------------------------------------------------------------------
    # Private — FinBERT inference
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load FinBERT model from HuggingFace."""
        try:
            from transformers import pipeline
            logger.info(f"[finbert] Loading {self.model_name}...")
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,   # CPU — set to 0 for GPU
            )
            self._loaded = True
            logger.info("[finbert] Model loaded successfully.")
        except ImportError:
            logger.warning(
                "[finbert] transformers not installed. "
                "pip install transformers. Falling back to keyword scoring."
            )
            self._loaded = False
        except Exception as e:
            logger.warning(f"[finbert] Failed to load model: {e}. Falling back.")
            self._loaded = False

    def _finbert_score(self, headline: str) -> float:
        """Score single headline with FinBERT."""
        try:
            result = self._pipeline(
                headline[:self.max_length],
                truncation=True,
            )[0]
            label = result["label"].lower()
            score = result["score"]
            if label == "positive":
                return float(score)
            elif label == "negative":
                return float(-score)
            else:
                return 0.0
        except Exception:
            return self._keyword_score(headline)

    def _finbert_batch(self, headlines: list[str]) -> list[float]:
        """Score batch of headlines with FinBERT."""
        try:
            truncated = [h[:self.max_length] for h in headlines]
            results   = self._pipeline(
                truncated,
                batch_size=self.batch_size,
                truncation=True,
            )
            scores = []
            for r in results:
                label = r["label"].lower()
                score = r["score"]
                if label == "positive":
                    scores.append(float(score))
                elif label == "negative":
                    scores.append(float(-score))
                else:
                    scores.append(0.0)
            return scores
        except Exception as e:
            logger.warning(f"[finbert] Batch scoring failed: {e}")
            return [self._keyword_score(h) for h in headlines]

    # -----------------------------------------------------------------------
    # Private — keyword fallback
    # -----------------------------------------------------------------------

    def _keyword_score(self, headline: str) -> float:
        """
        Simple keyword-based sentiment scoring.
        Used when FinBERT is not available.
        """
        if not headline:
            return 0.0
        words = set(headline.lower().split())
        pos   = len(words & self.POSITIVE_WORDS)
        neg   = len(words & self.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total