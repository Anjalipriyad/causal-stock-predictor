"""
significance.py
---------------
Statistical significance tests for the paper's central empirical claims.

Every directional accuracy comparison in the paper needs a p-value.
Without statistical testing, differences of 0.54 vs 0.51 are noise —
not publishable results.

Tests implemented:
    1. McNemar's test     — paired comparison of two models' correct/incorrect
                            calls on the same test set. Use when comparing
                            pcmci_causal vs all_features on the same data.

    2. Bootstrap CI       — non-parametric confidence interval for a single
                            model's directional accuracy. Avoids normality
                            assumption (financial returns are fat-tailed).

    3. Binomial test      — tests if DA is significantly > 0.50 (random baseline).
                            Simplest test; use for the "is this model better
                            than random?" question.

    4. DM test            — Diebold-Mariano test for equal predictive accuracy
                            on return RMSE. Use when comparing models on the
                            regression task, not just direction.

Paper usage:
    Table 2 should report: DA ± CI, p-value vs random (binomial),
    and p-value for causal vs all_features comparison (McNemar).

Usage:
    from ml.src.evaluation.significance import SignificanceTester
    tester = SignificanceTester()

    # Is DA > 0.50?
    result = tester.binomial_test(da=0.543, n=252)

    # Is model A better than model B?
    result = tester.mcnemar_test(pred_a, pred_b, y_true)

    # Bootstrap CI for DA
    low, high = tester.bootstrap_da_ci(y_pred, y_true)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SignificanceResult:
    test_name:   str
    statistic:   float
    p_value:     float
    significant: bool           # p < 0.05
    effect_size: Optional[float] = None
    ci_low:      Optional[float] = None
    ci_high:     Optional[float] = None
    n:           Optional[int]  = None
    note:        str            = ""

    def __str__(self) -> str:
        sig_str = "✓ SIGNIFICANT" if self.significant else "✗ not significant"
        parts = [
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f} → {sig_str}"
        ]
        if self.ci_low is not None and self.ci_high is not None:
            parts.append(f"  95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}]")
        if self.effect_size is not None:
            parts.append(f"  Effect size: {self.effect_size:.4f}")
        if self.note:
            parts.append(f"  Note: {self.note}")
        return "\n".join(parts)


class SignificanceTester:
    """
    Statistical significance tests for model comparison.
    All tests are two-sided unless stated otherwise.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # -----------------------------------------------------------------------
    # Test 1: Binomial test — is DA > 0.50?
    # -----------------------------------------------------------------------

    def binomial_test(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
        null_da: float = 0.50,
    ) -> SignificanceResult:
        """
        One-sided binomial test: H0: DA ≤ null_da (random baseline).
        HA: DA > null_da.

        This is the most basic question: is the model better than flipping
        a coin? Use this for every model in Table 2.

        Args:
            y_pred:   Predicted returns (sign determines predicted direction)
            y_true:   Actual returns
            null_da:  Null hypothesis DA (default 0.50 = random)

        Returns:
            SignificanceResult with p-value for one-sided test.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        n        = len(y_true)
        correct  = int(((y_pred >= 0) == (y_true >= 0)).sum())
        da       = correct / n

        # Binomial test: probability of getting >= correct out of n with p=null_da
        result = stats.binomtest(correct, n, null_da, alternative="greater")

        return SignificanceResult(
            test_name   = f"Binomial test (H0: DA ≤ {null_da})",
            statistic   = da,
            p_value     = result.pvalue,
            significant = result.pvalue < self.alpha,
            n           = n,
            note        = f"Observed {correct}/{n} correct ({da:.3f})",
        )

    # -----------------------------------------------------------------------
    # Test 2: McNemar's test — is model A better than model B?
    # -----------------------------------------------------------------------

    def mcnemar_test(
        self,
        y_pred_a: pd.Series,
        y_pred_b: pd.Series,
        y_true:   pd.Series,
        label_a:  str = "model_A",
        label_b:  str = "model_B",
    ) -> SignificanceResult:
        """
        McNemar's test for paired directional accuracy comparison.

        Tests H0: model A and model B make the same number of directional
        errors (i.e., their DAs are equal). This is the RIGHT test for
        comparing two models on the same test set — it accounts for the
        fact that both models see the same market conditions.

        DO NOT use a two-sample proportion test here. McNemar's is correct
        because the predictions are paired (same dates).

        Args:
            y_pred_a: Predicted returns from model A
            y_pred_b: Predicted returns from model B
            y_true:   Actual returns (same for both)
            label_a:  Name for model A (for reporting)
            label_b:  Name for model B (for reporting)

        Returns:
            SignificanceResult. p < 0.05 means models differ significantly.
        """
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

        y_pred_a, y_true = self._align(y_pred_a, y_true)
        y_pred_b, _      = self._align(y_pred_b, y_true)

        correct_a = (np.sign(y_pred_a) == np.sign(y_true)).astype(int)
        correct_b = (np.sign(y_pred_b) == np.sign(y_true)).astype(int)

        # Contingency table:
        #           B correct   B wrong
        # A correct    n11         n10
        # A wrong      n01         n00
        n11 = int((correct_a & correct_b).sum())
        n10 = int((correct_a & ~correct_b).sum())
        n01 = int((~correct_a & correct_b).sum())
        n00 = int((~correct_a & ~correct_b).sum())

        table   = np.array([[n11, n10], [n01, n00]])
        result  = sm_mcnemar(table, exact=False, correction=True)

        da_a = correct_a.mean()
        da_b = correct_b.mean()
        n    = len(y_true)

        return SignificanceResult(
            test_name   = f"McNemar's test ({label_a} vs {label_b})",
            statistic   = result.statistic,
            p_value     = result.pvalue,
            significant = result.pvalue < self.alpha,
            effect_size = da_a - da_b,
            n           = n,
            note        = (
                f"{label_a} DA={da_a:.3f}, {label_b} DA={da_b:.3f}. "
                f"Discordant pairs: {label_a}-only={n10}, {label_b}-only={n01}"
            ),
        )

    # -----------------------------------------------------------------------
    # Test 3: Bootstrap confidence interval for DA
    # -----------------------------------------------------------------------

    def bootstrap_da_ci(
        self,
        y_pred: pd.Series,
        y_true: pd.Series,
        n_bootstrap: int = 2000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> SignificanceResult:
        """
        Bootstrap confidence interval for directional accuracy.

        Financial returns are fat-tailed and autocorrelated. The normal
        approximation DA ± 1.96 * sqrt(DA*(1-DA)/n) is unreliable.
        Use bootstrap instead.

        The bootstrap resamples BLOCKS of consecutive dates (block bootstrap)
        to preserve autocorrelation structure. Block size = sqrt(n).

        Args:
            y_pred:      Predicted returns
            y_true:      Actual returns
            n_bootstrap: Number of bootstrap iterations (2000 is sufficient)
            ci:          Confidence level (0.95 = 95% CI)
            seed:        Random seed for reproducibility

        Returns:
            SignificanceResult with ci_low, ci_high.
        """
        y_pred, y_true = self._align(y_pred, y_true)
        n         = len(y_true)
        rng       = np.random.default_rng(seed)

        # Block bootstrap: preserve temporal autocorrelation
        block_size = max(1, int(np.sqrt(n)))
        correct    = ((y_pred >= 0) == (y_true >= 0)).astype(int).values
        observed_da = correct.mean()

        boot_das = []
        for _ in range(n_bootstrap):
            # Sample block start indices
            n_blocks = int(np.ceil(n / block_size))
            starts   = rng.integers(0, n - block_size + 1, size=n_blocks)
            indices  = np.concatenate([
                np.arange(s, min(s + block_size, n)) for s in starts
            ])[:n]
            boot_das.append(correct[indices].mean())

        boot_das = np.array(boot_das)
        alpha    = 1 - ci
        ci_low   = float(np.percentile(boot_das, 100 * alpha / 2))
        ci_high  = float(np.percentile(boot_das, 100 * (1 - alpha / 2)))

        # One-sided p-value: fraction of bootstrap DAs below 0.50
        p_value = float((boot_das <= 0.50).mean())

        return SignificanceResult(
            test_name   = f"Block Bootstrap DA CI (block_size={block_size})",
            statistic   = observed_da,
            p_value     = p_value,
            significant = ci_low > 0.50,      # CI entirely above random
            ci_low      = ci_low,
            ci_high     = ci_high,
            n           = n,
            note        = (
                f"Observed DA={observed_da:.4f}. "
                f"{int(ci*100)}% CI=[{ci_low:.4f}, {ci_high:.4f}]. "
                f"{'Above random baseline.' if ci_low > 0.50 else 'Overlaps random baseline.'}"
            ),
        )

    # -----------------------------------------------------------------------
    # Test 4: Diebold-Mariano test — equal predictive accuracy on returns
    # -----------------------------------------------------------------------

    def diebold_mariano_test(
        self,
        y_pred_a: pd.Series,
        y_pred_b: pd.Series,
        y_true:   pd.Series,
        h:        int = 5,
        label_a:  str = "model_A",
        label_b:  str = "model_B",
    ) -> SignificanceResult:
        """
        Diebold-Mariano (DM) test for equal predictive accuracy.

        Tests H0: models A and B have equal expected squared prediction error.
        HA: model A has lower MSE than model B (one-sided).

        Appropriate for comparing models on the REGRESSION task (RMSE/MSE).
        Uses HAC (heteroskedasticity and autocorrelation consistent) standard
        errors to handle serial correlation in h-step-ahead forecasts.

        Args:
            y_pred_a: Predicted returns from model A
            y_pred_b: Predicted returns from model B
            y_true:   Actual returns
            h:        Forecast horizon in periods (use config horizon_days)
            label_a:  Name for model A
            label_b:  Name for model B

        Returns:
            SignificanceResult. p < 0.05 means A significantly outperforms B.
        """
        y_pred_a, y_true = self._align(y_pred_a, y_true)
        y_pred_b, _      = self._align(y_pred_b, y_true)

        e_a = (y_true.values - y_pred_a.values) ** 2
        e_b = (y_true.values - y_pred_b.values) ** 2
        d   = e_a - e_b   # loss differential: negative = A is better

        n        = len(d)
        d_bar    = d.mean()

        # HAC variance estimator (Newey-West with h-1 lags)
        n_lags   = h - 1
        var_d    = self._newey_west_variance(d, n_lags)

        if var_d <= 0:
            return SignificanceResult(
                test_name="Diebold-Mariano test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                note="Variance of loss differential is zero — models are identical.",
            )

        dm_stat  = d_bar / np.sqrt(var_d / n)

        # Two-sided p-value (Harvey et al. 1997 small-sample correction)
        # Use t-distribution with n-1 degrees of freedom
        p_value  = 2 * float(stats.t.sf(abs(dm_stat), df=n - 1))

        rmse_a   = float(np.sqrt(e_a.mean()))
        rmse_b   = float(np.sqrt(e_b.mean()))

        return SignificanceResult(
            test_name   = f"Diebold-Mariano test ({label_a} vs {label_b})",
            statistic   = float(dm_stat),
            p_value     = p_value,
            significant = p_value < self.alpha,
            effect_size = rmse_a - rmse_b,  # negative = A is better
            n           = n,
            note        = (
                f"{label_a} RMSE={rmse_a:.5f}, {label_b} RMSE={rmse_b:.5f}. "
                f"DM stat={dm_stat:.3f}. "
                f"{'A significantly better.' if (p_value < self.alpha and dm_stat < 0) else 'No significant difference.'}"
            ),
        )

    # -----------------------------------------------------------------------
    # Batch: run all tests for a full regime comparison table
    # -----------------------------------------------------------------------

    def full_regime_significance_table(
        self,
        predictions: dict[str, dict[str, pd.Series]],
        actuals:     dict[str, pd.Series],
        horizon:     int = 5,
    ) -> pd.DataFrame:
        """
        Run significance tests for all model pairs across all regimes.
        This produces the statistical annex for Table 2 in the paper.

        Args:
            predictions: {model_name: {regime_name: predicted_returns_series}}
            actuals:     {regime_name: actual_returns_series}
            horizon:     Forecast horizon for DM test

        Returns:
            DataFrame with columns:
                regime, model_a, model_b, da_a, da_b, da_diff,
                mcnemar_p, dm_p, binomial_p_a, binomial_p_b,
                bootstrap_ci_a_low, bootstrap_ci_a_high
        """
        model_names = list(predictions.keys())
        rows = []

        for regime_name, y_true in actuals.items():
            # Binomial test for each model individually
            for model_name in model_names:
                if regime_name not in predictions[model_name]:
                    continue
                y_pred = predictions[model_name][regime_name]
                try:
                    binom  = self.binomial_test(y_pred, y_true)
                    boot   = self.bootstrap_da_ci(y_pred, y_true)
                    da     = float(binom.statistic)
                    rows.append({
                        "regime":         regime_name,
                        "model":          model_name,
                        "da":             da,
                        "binomial_p":     binom.p_value,
                        "ci_low":         boot.ci_low,
                        "ci_high":        boot.ci_high,
                        "sig_vs_random":  binom.significant,
                    })
                except Exception as e:
                    logger.warning(f"[significance] {model_name}/{regime_name}: {e}")

        individual_df = pd.DataFrame(rows)

        # McNemar + DM tests: all pairwise model comparisons per regime
        pair_rows = []
        for regime_name, y_true in actuals.items():
            for i, ma in enumerate(model_names):
                for mb in model_names[i+1:]:
                    if (regime_name not in predictions[ma] or
                            regime_name not in predictions[mb]):
                        continue
                    ya = predictions[ma][regime_name]
                    yb = predictions[mb][regime_name]
                    try:
                        mc = self.mcnemar_test(ya, yb, y_true, ma, mb)
                        dm = self.diebold_mariano_test(ya, yb, y_true, horizon, ma, mb)
                        pair_rows.append({
                            "regime":     regime_name,
                            "model_a":    ma,
                            "model_b":    mb,
                            "da_diff":    mc.effect_size,
                            "mcnemar_p":  mc.p_value,
                            "dm_p":       dm.p_value,
                            "mcnemar_sig": mc.significant,
                            "dm_sig":     dm.significant,
                        })
                    except Exception as e:
                        logger.warning(
                            f"[significance] {ma} vs {mb} / {regime_name}: {e}"
                        )

        pair_df = pd.DataFrame(pair_rows)
        logger.info(
            f"[significance] Significance table complete: "
            f"{len(individual_df)} individual tests, {len(pair_df)} pairwise tests."
        )
        return individual_df, pair_df

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _align(
        self, y_pred: pd.Series, y_true: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Align on common index, drop NaN, convert to Series."""
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred)
        if isinstance(y_true, np.ndarray):
            y_true = pd.Series(y_true)
        combined = pd.concat([y_pred, y_true], axis=1).dropna()
        return combined.iloc[:, 0], combined.iloc[:, 1]

    def _newey_west_variance(self, d: np.ndarray, n_lags: int) -> float:
        """
        Newey-West HAC variance estimator for loss differential series.
        Handles serial correlation up to n_lags lags.
        """
        n     = len(d)
        d_bar = d.mean()
        d_c   = d - d_bar

        # Start with variance term
        var = float(np.dot(d_c, d_c) / n)

        # Add covariance terms with Bartlett kernel weights
        for lag in range(1, n_lags + 1):
            weight   = 1 - lag / (n_lags + 1)   # Bartlett kernel
            cov_term = float(np.dot(d_c[lag:], d_c[:-lag]) / n)
            var     += 2 * weight * cov_term

        return max(var, 0.0)   # clip at 0 (can be negative in small samples)