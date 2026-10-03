# Mathematical specification and public-method references

This document states the implemented mathematics and the public references it follows.

## Static split conformal prediction

For n held-out nonnegative conformity scores, let k = ceil((n+1)(1-alpha)), with 0 <= alpha < 1. Sort scores and take the k-th smallest score. If k > n, use +Infinity. Empty calibration also returns +Infinity with explicit status.

For binary classification, s(p,0)=p and s(p,1)=1-p. Include a label exactly when its score is <= qHat. For regression, use absolute residuals from a fixed model and report [prediction-qHat,prediction+qHat].

The marginal coverage statement requires exchangeable calibration/test examples and a score rule fixed without using those calibration labels to train or select it. It is not a conditional probability about any particular answer. Ties can make coverage conservative. The package does not implement automatic handling of time series, distribution drift, data leakage, multiple testing, or groupwise coverage. It cannot certify exchangeability from input arrays.

Primary reference: Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, arXiv:2107.07511, especially Section 1 and Appendix D:
https://arxiv.org/html/2107.07511v6

Implementation note: the integer rank is selected directly, avoiding a second floating-point quantile calculation. The nearest-rank general quantile is not NumPy's default interpolated quantile.

## Mean and variance

Welford state contains n, mean, and m2 (sum of squared deviations). Update with delta=x-mean, mean'=mean+delta/(n+1), m2'=m2+delta*(x-mean'). Population variance is m2/n; sample variance is m2/(n-1). The latter requires n>=2. Finite arithmetic overflow is rejected. This is floating-point arithmetic, not an arbitrary-precision algorithm.

Original methodological reference: B. P. Welford (1962), *Note on a Method for Calculating Corrected Sums of Squares and Products*, Technometrics 4(3), 419-420, DOI 10.1080/00401706.1962.10490022. Numerical reference checks use NumPy var on fixed fixtures, not a claim that every input is exactly represented.

## Two-sample KS

The empirical statistic is max_x |F_A(x)-F_B(x)| and treats tied observations together. Its p-value helper evaluates the limiting Kolmogorov survival function at lambda=(sqrt(n*m/(n+m)) + .12 + .11/sqrt(n*m/(n+m)))*D. This common finite-size-adjusted asymptotic approximation is not an exact finite-n test.

For larger lambda, evaluate 2*sum((-1)^(k-1)*exp(-2*k*k*lambda*lambda)). For small lambda, use the complementary theta-series form to avoid cancellation. The core survival function is cross-checked against scipy.special.kolmogorov. Matching that function does NOT turn this into scipy.stats.ks_2samp's exact two-sample p-value.

Primary implementation documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kolmogorov.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

P-values assume independent continuous samples. Discreteness, ties, repeated testing and temporal dependence need separate treatment. A large p-value does not prove no drift; a small one does not establish the cause of a difference.

## Binary diagnostics

Brier loss for one observed binary outcome is (p-y)^2. ECE here bins positive-class probabilities in equal-width bins and sums weighted absolute probability/outcome discrepancies. This is deliberately narrower than a generic calibration metric claim.

Primary context and estimator distinctions:
https://proceedings.mlr.press/v70/guo17a.html
https://arxiv.org/abs/2109.03480

## Scope

No adaptive or online significance updates, learned policies, or automatic remediation are included. The software measures outputs; it does not decide what action to take. Function-level public-literature support is not a freedom-to-operate opinion.
