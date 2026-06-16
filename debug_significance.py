
import pandas as pd
import numpy as np
from ml.src.evaluation.significance import SignificanceTester

# 1. Create dummy data based on NIFTY results (DA ~ 0.54)
# Let's assume N = 500 (approx test set size for 3500 rows)
n = 528 # Based on typical NIFTY test split
da = 0.543 # Observed in previous runs

correct = int(n * da)
incorrect = n - correct

y_true = pd.Series([1] * n)
y_pred = pd.Series([1] * correct + [-1] * incorrect)

tester = SignificanceTester()

print("="*60)
print("SIGNIFICANCE TEST WITH OVERLAP CORRECTION (HORIZON=5)")
print("="*60)

# Run binomial test with overlap correction
binom = tester.binomial_test(y_pred, y_true, horizon_days=5)
print(binom)

print("\n" + "="*60)
print("BOOTSTRAP CI (BLOCK BOOTSTRAP)")
print("="*60)
boot = tester.bootstrap_da_ci(y_pred, y_true, n_bootstrap=1000)
print(boot)
