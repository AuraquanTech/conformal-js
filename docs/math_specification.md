# conformal-js: Mathematical Specification

Based on reverse-engineering analysis by Bush (2026).

## 1. Conformal Classification

### Nonconformity Score
For a probabilistic classifier outputting p̂ ∈ [0,1]:

    s(x, y) = { 1 - p̂   if y = 1
               { p̂       if y = 0

### Prediction Set Construction
Given calibration scores S = {s₁, ..., sₙ} and current α:

    q̂ = Quantile(S, ⌈(n+1)(1-α)⌉/n)    [finite-sample correction]

    C(x) = { y ∈ {0,1} : s(x,y) ≤ q̂ }

Equivalently:
    Include y=1 if (1 - p̂) ≤ q̂
    Include y=0 if p̂ ≤ q̂

### ACI Update (Gibbs & Candès 2021)
After observing outcome yₜ:

    errₜ = 𝟙{yₜ ∉ C(xₜ)}
    αₜ₊₁ = clamp(αₜ + γ(α_target - errₜ), α_min, α_max)

where γ is the step size (default 0.005).

**Coverage guarantee**: Under exchangeability, E[errₜ] → α_target as t → ∞.
Under distribution shift, ACI provides adaptive guarantees (Theorem 1, arXiv:2106.00170).

## 2. Conformal Regression

### Prediction Interval
    [ŷ - q̂, ŷ + q̂]

where q̂ is the adaptive quantile of absolute residuals |yᵢ - ŷᵢ|.

### Regression ACI Update
    errₜ = 𝟙{yₜ ∉ [ŷₜ - q̂ₜ, ŷₜ + q̂ₜ]}
    Same α update as classification.

## 3. Kolmogorov-Smirnov Two-Sample Test

### Statistic
    D = sup_x |F̂_A(x) - F̂_B(x)|

where F̂ are empirical CDFs.

### Asymptotic p-value (Marsaglia formula)
    λ = (√(nm/(n+m)) + 0.12 + 0.11/√(nm/(n+m))) · D
    p ≈ 2·Σ_{k=1}^{∞} (-1)^{k-1} · exp(-2k²λ²)

## 4. Welford Online Variance (1962)

For streaming values x₁, x₂, ...:
    nₜ = nₜ₋₁ + 1
    δ = xₜ - μₜ₋₁
    μₜ = μₜ₋₁ + δ/nₜ
    M₂,ₜ = M₂,ₜ₋₁ + δ·(xₜ - μₜ)
    σ²ₜ = M₂,ₜ / nₜ

## 5. Expected Calibration Error (ECE)

    ECE = Σ_{b=1}^{B} (nᵦ/N) · |avg(p̂ᵦ) - avg(yᵦ)|

with B = 10 equal-width bins over [0, 1].

## 6. Brier Score

    BS = (p̂ - y)²

## 7. Bernoulli Nonconformity

    s(x, y=1) = 1 - p̂
    s(x, y=0) = p̂

Score ∈ [0, 1]. Lower = better calibrated.

## References

1. Gibbs & Candès (2021). "Adaptive Conformal Inference Under Distribution Shift". arXiv:2106.00170.
2. Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random World". Springer.
3. Angelopoulos & Bates (2021). "A Gentle Introduction to Conformal Prediction". arXiv:2107.07511.
4. Welford (1962). "Note on a Method for Calculating Corrected Sums of Squares". Technometrics 4(3).
5. Marsaglia, Tsang & Wang (2003). "Evaluating Kolmogorov's Distribution". JSS 8(18).
