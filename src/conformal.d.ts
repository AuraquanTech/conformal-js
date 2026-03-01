// conformal.d.ts — Type declarations for conformal-js
// Distribution-free uncertainty quantification for JavaScript/TypeScript

// ── Core Conformal ──────────────────────────────────────────

/** Empirical quantile using nearest-rank method */
export function quantile(scores: number[], q: number): number;

/** Split conformal calibration with finite-sample correction */
export function calibrate(calibScores: number[], alpha: number): number;

/** Conformal prediction set for binary classification */
export function conformalSet(
  pHat: number,
  qHat: number,
): { include0: boolean; include1: boolean };

/** Conformal prediction interval for regression */
export function conformalInterval(yHat: number, qHat: number): [number, number];

// ── ACI (Adaptive Conformal Inference) ──────────────────────

export interface ACIOptions {
  targetAlpha?: number;
  gamma?: number;
  alphaMin?: number;
  alphaMax?: number;
  windowSize?: number;
  warmStartN?: number;
}

export interface ACIState {
  alpha: number;
  targetAlpha: number;
  gamma: number;
  alphaMin: number;
  alphaMax: number;
  scores: number[];
  windowSize: number;
  warmStartN: number;
  t: number;
}

export interface ClassificationPrediction {
  qHat: number;
  include0: boolean;
  include1: boolean;
  n: number;
}

export interface RegressionPrediction {
  qHat: number;
  lower: number;
  upper: number;
  n: number;
}

/** Create an ACI state for online conformal inference */
export function createACI(options?: ACIOptions): ACIState;

/** Get prediction set from current ACI state (classification) */
export function aciPredict(
  state: ACIState,
  score: number,
  type?: "classification",
): ClassificationPrediction;

/** Get prediction interval from current ACI state (regression) */
export function aciPredict(
  state: ACIState,
  score: number,
  type: "regression",
): RegressionPrediction;

/** Update ACI state after observing outcome (Gibbs & Candès 2021) */
export function aciUpdate(
  state: ACIState,
  nonconformityScore: number,
  covered: boolean,
): ACIState;

// ── Welford Online Statistics ───────────────────────────────

export interface WelfordState {
  n: number;
  mean: number;
  m2: number;
}

export interface WelfordStats {
  mean: number;
  variance: number;
  std: number;
  n: number;
}

/** Create a Welford online statistics accumulator */
export function createWelford(): WelfordState;

/** Update accumulator with new observation (Welford 1962) */
export function welfordUpdate(state: WelfordState, x: number): WelfordState;

/** Get current mean, variance, std, n from accumulator */
export function welfordStats(state: WelfordState): WelfordStats;

// ── KS Test ─────────────────────────────────────────────────

/** Two-sample Kolmogorov-Smirnov D-statistic */
export function ksStatistic(sampleA: number[], sampleB: number[]): number;

/** Asymptotic p-value for two-sample KS test (Marsaglia approximation) */
export function ksPValue(D: number, n: number, m: number): number;

// ── Scoring ─────────────────────────────────────────────────

/** Bernoulli nonconformity score for binary classification */
export function bernoulliNonconformity(pHat: number, y: 0 | 1): number;

/** Brier score: (p̂ - y)² */
export function brierScore(pHat: number, y: number): number;

/** Expected Calibration Error with equal-width bins */
export function expectedCalibrationError(
  pHats: number[],
  ys: number[],
  numBins?: number,
): number;

// ── Utilities ───────────────────────────────────────────────

/** Clamp value between lo and hi */
export function clamp(x: number, lo: number, hi: number): number;

// ── Default Export ──────────────────────────────────────────

declare const _default: {
  quantile: typeof quantile;
  calibrate: typeof calibrate;
  conformalSet: typeof conformalSet;
  conformalInterval: typeof conformalInterval;
  createACI: typeof createACI;
  aciPredict: typeof aciPredict;
  aciUpdate: typeof aciUpdate;
  createWelford: typeof createWelford;
  welfordUpdate: typeof welfordUpdate;
  welfordStats: typeof welfordStats;
  ksStatistic: typeof ksStatistic;
  ksPValue: typeof ksPValue;
  bernoulliNonconformity: typeof bernoulliNonconformity;
  brierScore: typeof brierScore;
  expectedCalibrationError: typeof expectedCalibrationError;
  clamp: typeof clamp;
};
export default _default;
