/** Supported inputs. Readonly arrays and numeric typed arrays are not mutated. */
export type NumericArray = readonly number[] | Float32Array | Float64Array | Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array;
export interface CalibrationSummary {
  sampleSize: number;
  alpha: number;
  rank: number;
  qHat: number;
  status: 'finite' | 'insufficient-data' | 'empty-calibration' | 'full-coverage-request';
}
export interface Calibrator {
  readonly sampleSize: number;
  threshold(alpha: number): number;
  summary(alpha: number): CalibrationSummary;
}
export interface BinarySet { include0: boolean; include1: boolean; }
export interface WelfordState { n: number; mean: number; m2: number; }
export interface WelfordStats { mean: number; variance: number; std: number; n: number; }
/** Nearest-rank empirical quantile, q in [0,1], nonempty finite input. */
export function quantile(scores: NumericArray, q: number): number;
/** Finite nonnegative scores, alpha in [0,1). May return Infinity. */
export function calibrate(calibScores: NumericArray, alpha: number): number;
/** Immutable static calibration snapshot; does not update itself. */
export function createCalibrator(calibScores: NumericArray): Readonly<Calibrator>;
export function conformalSet(pHat: number, qHat: number): BinarySet;
/** qHat is a nonnegative absolute-residual threshold; Infinity is permitted. */
export function conformalInterval(yHat: number, qHat: number): [number, number];
export function createWelford(): WelfordState;
/** The only public API that mutates a supplied object. */
export function welfordUpdate(state: WelfordState, x: number): WelfordState;
export function welfordStats(state: WelfordState, options?: {ddof?: 0 | 1}): WelfordStats;
export function ksStatistic(sampleA: NumericArray, sampleB: NumericArray): number;
/** Approximation for independent continuous samples, not an exact test. */
export function ksPValue(D: number, n: number, m: number): number;
export function bernoulliNonconformity(pHat: number, y: 0 | 1): number;
export function brierScore(pHat: number, y: 0 | 1): number;
/** Positive-class binary ECE, not multiclass or top-label ECE. */
export function expectedCalibrationError(pHats: NumericArray, ys: NumericArray, numBins?: number): number;
export function clamp(x: number, lo: number, hi: number): number;
declare const api: {
  quantile: typeof quantile; calibrate: typeof calibrate; createCalibrator: typeof createCalibrator;
  conformalSet: typeof conformalSet; conformalInterval: typeof conformalInterval;
  createWelford: typeof createWelford; welfordUpdate: typeof welfordUpdate; welfordStats: typeof welfordStats;
  ksStatistic: typeof ksStatistic; ksPValue: typeof ksPValue;
  bernoulliNonconformity: typeof bernoulliNonconformity; brierScore: typeof brierScore;
  expectedCalibrationError: typeof expectedCalibrationError; clamp: typeof clamp;
};
export default api;
