// conformal.js — Distribution-free uncertainty quantification for JavaScript
// The first conformal prediction library for the JS/TS ecosystem.
// License: MIT
//
// Implements:
//   - Split Conformal Prediction (Vovk et al. 2005)
//   - Adaptive Conformal Inference / ACI (Gibbs & Candès 2021, arXiv:2106.00170)
//   - Welford Online Statistics (Welford 1962)
//   - Kolmogorov-Smirnov Two-Sample Drift Detection
//   - Bernoulli Nonconformity Scoring
//
// Zero dependencies. Works in browsers, Node.js, Deno, Bun, and service workers.

// ============================================================
// CORE: Split Conformal Prediction
// ============================================================

/**
 * Compute empirical quantile using nearest-rank method.
 * @param {number[]} scores - Nonconformity scores from calibration set
 * @param {number} q - Quantile level in [0, 1]
 * @returns {number} The q-th quantile of scores
 */
export function quantile(scores, q) {
  const n = scores.length;
  if (n === 0) return Infinity;
  const sorted = scores.slice().sort((a, b) => a - b);
  const idx = Math.min(Math.max(Math.ceil(q * n) - 1, 0), n - 1);
  return sorted[idx];
}

/**
 * Split conformal calibration: compute qHat from calibration residuals.
 * Includes finite-sample correction: ceil((n+1)(1-alpha)) / n.
 * @param {number[]} calibScores - Nonconformity scores on calibration data
 * @param {number} alpha - Significance level (e.g., 0.1 for 90% coverage)
 * @returns {number} Calibration quantile qHat
 */
export function calibrate(calibScores, alpha) {
  const n = calibScores.length;
  if (n === 0) return Infinity;
  const level = Math.ceil((n + 1) * (1 - alpha)) / n;
  return quantile(calibScores, Math.min(level, 1.0));
}

/**
 * Compute conformal prediction set for binary classification.
 * Given p̂ = P(Y=1), includes class y if nonconformity(y) ≤ qHat.
 * @param {number} pHat - Predicted probability for class 1
 * @param {number} qHat - Calibration quantile threshold
 * @returns {{include0: boolean, include1: boolean}} Which classes are in the set
 */
export function conformalSet(pHat, qHat) {
  return {
    include0: pHat <= qHat,        // nonconformity(y=0) = pHat
    include1: (1 - pHat) <= qHat,  // nonconformity(y=1) = 1 - pHat
  };
}

/**
 * Compute conformal prediction interval for regression.
 * @param {number} yHat - Point prediction
 * @param {number} qHat - Calibration quantile of |residuals|
 * @returns {[number, number]} [lower, upper] prediction interval
 */
export function conformalInterval(yHat, qHat) {
  return [yHat - qHat, yHat + qHat];
}

// ============================================================
// ACI: Adaptive Conformal Inference (Gibbs & Candès 2021)
// ============================================================

/**
 * Create an ACI state object for online conformal inference.
 *
 * ACI adapts the significance level α over time based on observed
 * coverage errors, providing distribution-shift-robust guarantees.
 *
 * @param {Object} [options]
 * @param {number} [options.targetAlpha=0.1] - Target miscoverage rate
 * @param {number} [options.gamma=0.005] - Step size for α updates
 * @param {number} [options.alphaMin=0.001] - Minimum α (floor)
 * @param {number} [options.alphaMax=0.5] - Maximum α (ceiling)
 * @param {number} [options.windowSize=2000] - Rolling score window size
 * @param {number} [options.warmStartN=50] - Min observations before real predictions
 * @returns {ACIState} Initialized ACI state
 */
export function createACI(options = {}) {
  const {
    targetAlpha = 0.1,
    gamma = 0.005,
    alphaMin = 0.001,
    alphaMax = 0.5,
    windowSize = 2000,
    warmStartN = 50,
  } = options;

  return {
    alpha: targetAlpha,
    targetAlpha,
    gamma,
    alphaMin,
    alphaMax,
    scores: [],
    windowSize,
    warmStartN,
    t: 0,
  };
}

/**
 * Get current prediction set/interval from ACI state.
 *
 * During warm-start (fewer than warmStartN observations), returns
 * maximally conservative predictions (full set or infinite interval).
 *
 * @param {ACIState} state - ACI state from createACI()
 * @param {number} score - Current p̂ (classification) or point prediction (regression)
 * @param {'classification'|'regression'} [type='classification']
 * @returns {ClassificationPrediction|RegressionPrediction}
 */
export function aciPredict(state, score, type = "classification") {
  const isWarm = state.scores.length >= state.warmStartN;
  const qHat = isWarm
    ? quantile(state.scores, 1 - state.alpha)
    : type === "classification"
      ? 1.0
      : Infinity;

  if (type === "classification") {
    return { qHat, ...conformalSet(score, qHat), n: state.scores.length };
  }
  const [lower, upper] = conformalInterval(score, qHat);
  return { qHat, lower, upper, n: state.scores.length };
}

/**
 * Update ACI state after observing true outcome.
 *
 * Implements Algorithm 1 from Gibbs & Candès (2021), arXiv:2106.00170:
 *   errₜ = 𝟙{yₜ ∉ Cₜ(xₜ)}
 *   αₜ₊₁ = clamp(αₜ + γ(α_target - errₜ), α_min, α_max)
 *
 * Coverage guarantee: Under exchangeability, E[errₜ] → α_target as t → ∞.
 *
 * @param {ACIState} state - ACI state (mutated in place)
 * @param {number} nonconformityScore - Realized nonconformity score
 * @param {boolean} covered - Whether the true label was in the prediction set
 * @returns {ACIState} Updated state with new alpha
 */
export function aciUpdate(state, nonconformityScore, covered) {
  const err = covered ? 0 : 1;

  // α_{t+1} = clamp(α_t + γ(α_target - err_t), α_min, α_max)
  state.alpha = clamp(
    state.alpha + state.gamma * (state.targetAlpha - err),
    state.alphaMin,
    state.alphaMax,
  );

  // Rolling window of nonconformity scores for calibration
  state.scores.push(nonconformityScore);
  if (state.scores.length > state.windowSize) {
    state.scores.splice(0, state.scores.length - state.windowSize);
  }

  state.t += 1;
  return state;
}

// ============================================================
// WELFORD: Online Mean/Variance (Welford 1962)
// ============================================================

/**
 * Create a Welford online statistics accumulator.
 * Numerically stable single-pass algorithm for mean and variance.
 * @returns {WelfordState}
 */
export function createWelford() {
  return { n: 0, mean: 0, m2: 0 };
}

/**
 * Update Welford accumulator with a new observation.
 *
 * Uses the numerically stable recurrence from Welford (1962):
 *   δ  = xₜ - μₜ₋₁
 *   μₜ = μₜ₋₁ + δ/nₜ
 *   M₂ = M₂ + δ·(xₜ - μₜ)
 *
 * @param {WelfordState} state - Welford state (mutated in place)
 * @param {number} x - New observation
 * @returns {WelfordState} Updated state
 */
export function welfordUpdate(state, x) {
  state.n += 1;
  const delta = x - state.mean;
  state.mean += delta / state.n;
  const delta2 = x - state.mean;
  state.m2 += delta * delta2;
  return state;
}

/**
 * Get current statistics from Welford accumulator.
 * @param {WelfordState} state
 * @returns {{mean: number, variance: number, std: number, n: number}}
 */
export function welfordStats(state) {
  if (state.n < 2) return { mean: state.mean, variance: 0, std: 0, n: state.n };
  const variance = state.m2 / state.n;
  return { mean: state.mean, variance, std: Math.sqrt(variance), n: state.n };
}

// ============================================================
// KS TEST: Kolmogorov-Smirnov Two-Sample Drift Detection
// ============================================================

/**
 * Compute two-sample Kolmogorov-Smirnov D-statistic.
 *   D = sup_x |F̂_A(x) - F̂_B(x)|
 *
 * @param {number[]} sampleA - First sample
 * @param {number[]} sampleB - Second sample
 * @returns {number} KS D-statistic in [0, 1]
 */
export function ksStatistic(sampleA, sampleB) {
  const a = sampleA.slice().sort((x, y) => x - y);
  const b = sampleB.slice().sort((x, y) => x - y);
  const n = a.length;
  const m = b.length;
  if (n === 0 || m === 0) return 0;

  let i = 0;
  let j = 0;
  let D = 0;
  while (i < n || j < m) {
    const x = i < n && (j >= m || a[i] <= b[j]) ? a[i] : b[j];
    while (i < n && a[i] <= x) i++;
    while (j < m && b[j] <= x) j++;
    D = Math.max(D, Math.abs(i / n - j / m));
  }
  return D;
}

/**
 * Asymptotic p-value for two-sample KS test.
 *
 * Uses the Marsaglia approximation:
 *   λ = (√(nm/(n+m)) + 0.12 + 0.11/√(nm/(n+m))) · D
 *   p ≈ 2·Σ_{k=1}^{∞} (-1)^{k-1} · exp(-2k²λ²)
 *
 * @param {number} D - KS D-statistic
 * @param {number} n - Size of first sample
 * @param {number} m - Size of second sample
 * @returns {number} Approximate p-value in [0, 1]
 */
export function ksPValue(D, n, m) {
  if (D === 0 || n === 0 || m === 0) return 1.0;
  const en = Math.sqrt((n * m) / (n + m));
  const lambda = (en + 0.12 + 0.11 / en) * D;
  let s = 0;
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda);
    s += k % 2 === 1 ? term : -term;
    if (term < 1e-10) break;
  }
  return Math.max(0, Math.min(1, 2 * s));
}

// ============================================================
// SCORING
// ============================================================

/**
 * Compute Bernoulli nonconformity score for binary classification.
 *   s(x, y=1) = 1 - p̂
 *   s(x, y=0) = p̂
 *
 * @param {number} pHat - Predicted P(Y=1) in [0, 1]
 * @param {0|1} y - True label
 * @returns {number} Nonconformity score in [0, 1]
 */
export function bernoulliNonconformity(pHat, y) {
  return y === 1 ? 1 - pHat : pHat;
}

/**
 * Compute Brier score for a single prediction.
 *   BS = (p̂ - y)²
 *
 * @param {number} pHat - Predicted probability
 * @param {number} y - True outcome (0 or 1)
 * @returns {number} Brier score in [0, 1]
 */
export function brierScore(pHat, y) {
  return (pHat - y) * (pHat - y);
}

/**
 * Compute Expected Calibration Error (ECE) from binned predictions.
 *   ECE = Σ (n_b / N) · |avg(p̂_b) - avg(y_b)|
 *
 * @param {number[]} pHats - Predicted probabilities
 * @param {number[]} ys - True outcomes (0 or 1)
 * @param {number} [numBins=10] - Number of equal-width bins
 * @returns {number} ECE in [0, 1]
 */
export function expectedCalibrationError(pHats, ys, numBins = 10) {
  if (pHats.length === 0) return 0;
  const N = pHats.length;
  let ece = 0;

  for (let b = 0; b < numBins; b++) {
    const lo = b / numBins;
    const hi = (b + 1) / numBins;
    let sumP = 0;
    let sumY = 0;
    let count = 0;

    for (let i = 0; i < N; i++) {
      if (pHats[i] >= lo && (pHats[i] < hi || (b === numBins - 1 && pHats[i] <= hi))) {
        sumP += pHats[i];
        sumY += ys[i];
        count++;
      }
    }

    if (count > 0) {
      ece += (count / N) * Math.abs(sumP / count - sumY / count);
    }
  }

  return ece;
}

// ============================================================
// UTILITIES
// ============================================================

/**
 * Clamp a value between lo and hi.
 * @param {number} x
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Default export for convenience
export default {
  // Core conformal
  quantile,
  calibrate,
  conformalSet,
  conformalInterval,
  // ACI
  createACI,
  aciPredict,
  aciUpdate,
  // Welford
  createWelford,
  welfordUpdate,
  welfordStats,
  // KS Test
  ksStatistic,
  ksPValue,
  // Scoring
  bernoulliNonconformity,
  brierScore,
  expectedCalibrationError,
  // Utilities
  clamp,
};
