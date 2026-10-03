// conformal-js. MIT License; see LICENSE.
// Published-method references and assumptions: docs/MATHEMATICS.md.
// No filesystem, network, model-training, or adaptive-control side effects.

/** @typedef {readonly number[] | Float32Array | Float64Array | Int8Array | Uint8Array | Uint8ClampedArray | Int16Array | Uint16Array | Int32Array | Uint32Array} NumericArray */
/** @typedef {{n:number,mean:number,m2:number}} WelfordState */
/** @typedef {{sampleSize:number,alpha:number,rank:number,qHat:number,status:'finite'|'insufficient-data'|'empty-calibration'|'full-coverage-request'}} CalibrationSummary */

const MAX_BINS = 65536;

/** @param {unknown} x @param {string} name @returns {asserts x is number} */
function finite(x, name) {
  if (typeof x !== 'number') throw new TypeError(`${name} must be a number`);
  if (!Number.isFinite(x)) throw new RangeError(`${name} must be finite`);
}
/** @param {number} x @param {string} name */
function probability(x, name) {
  finite(x, name);
  if (x < 0 || x > 1) throw new RangeError(`${name} must be between 0 and 1`);
}
/** @param {number} alpha */
function checkAlpha(alpha) {
  finite(alpha, 'alpha');
  if (alpha < 0 || alpha >= 1) throw new RangeError('alpha must be in [0, 1)');
}
/** @param {number} y @param {string} name */
function binary(y, name) {
  if (y !== 0 && y !== 1) throw new RangeError(`${name} must be 0 or 1`);
}
/** @param {unknown} xs @param {string} name @returns {asserts xs is NumericArray} */
function arrayType(xs, name) {
  if (!Array.isArray(xs) && !(ArrayBuffer.isView(xs) && Object.prototype.toString.call(xs) !== '[object DataView]')) {
    throw new TypeError(`${name} must be an array or numeric typed array`);
  }
  // Reject BigInt arrays even when empty, and concurrent shared-buffer inputs.
  if (!Array.isArray(xs)) {
    const tag = Object.prototype.toString.call(xs);
    if (tag === '[object BigInt64Array]' || tag === '[object BigUint64Array]') {
      throw new TypeError(`${name} must contain numbers, not bigint`);
    }
    if (Object.prototype.toString.call(xs.buffer) === '[object SharedArrayBuffer]') {
      throw new TypeError(`${name} must not use a shared buffer`);
    }
  }
}
/** @param {NumericArray} xs @param {string} name @param {boolean} [nonnegative] @returns {number[]} */
function sortedCopy(xs, name, nonnegative = false) {
  arrayType(xs, name);
  const copy = new Array(xs.length);
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    if (typeof x !== 'number' || !Number.isFinite(x)) finite(x, `${name}[${i}]`);
    if (nonnegative && x < 0) throw new RangeError(`${name}[${i}] must be nonnegative`);
    copy[i] = x;
  }
  return copy.sort((a, b) => a - b);
}
/** @param {number} n @param {number} alpha */
function rank(n, alpha) {
  // Select the integer order statistic directly; no second quantile rounding.
  return Math.ceil((n + 1) * (1 - alpha));
}
/** @param {number[]} sorted @param {number} alpha */
function threshold(sorted, alpha) {
  const k = rank(sorted.length, alpha);
  return k > sorted.length ? Infinity : sorted[k - 1];
}
/** @param {number} qHat */
function checkRadius(qHat) {
  if (typeof qHat !== 'number') throw new TypeError('qHat must be a number');
  if (Number.isNaN(qHat) || qHat < 0) throw new RangeError('qHat must be nonnegative (Infinity is allowed)');
}

/**
 * Nearest-rank empirical quantile. q=0 returns the minimum; q=1 the maximum.
 * Empty input is an error. The input is never mutated.
 * @param {NumericArray} scores @param {number} q @returns {number}
 */
export function quantile(scores, q) {
  probability(q, 'q');
  const sorted = sortedCopy(scores, 'scores');
  if (!sorted.length) throw new RangeError('scores must not be empty');
  return sorted[Math.max(0, Math.ceil(q * sorted.length) - 1)];
}

/**
 * Split conformal threshold for finite, nonnegative calibration scores.
 * Returns Infinity when n=0, alpha=0, or ceil((n+1)(1-alpha)) > n.
 * Coverage is marginal under exchangeability, not confidence in each answer.
 * @param {NumericArray} calibScores @param {number} alpha @returns {number}
 */
export function calibrate(calibScores, alpha) {
  checkAlpha(alpha);
  return threshold(sortedCopy(calibScores, 'calibScores', true), alpha);
}

/**
 * Validate and sort once, then query many alpha levels in O(1) each.
 * Holds a private snapshot: changing the caller's array cannot change results.
 * This is static calibration, not an online updating model.
 * @param {NumericArray} calibScores
 * @returns {Readonly<{sampleSize:number,threshold:(alpha:number)=>number,summary:(alpha:number)=>CalibrationSummary}>}
 */
export function createCalibrator(calibScores) {
  const sorted = sortedCopy(calibScores, 'calibScores', true);
  return Object.freeze({
    sampleSize: sorted.length,
    threshold(alpha) {
      checkAlpha(alpha);
      return threshold(sorted, alpha);
    },
    summary(alpha) {
      checkAlpha(alpha);
      const k = rank(sorted.length, alpha);
      const qHat = threshold(sorted, alpha);
      const status = !sorted.length ? 'empty-calibration' : alpha === 0
        ? 'full-coverage-request' : k > sorted.length ? 'insufficient-data' : 'finite';
      return { sampleSize: sorted.length, alpha, rank: k, qHat, status };
    },
  });
}

/**
 * Binary prediction set from P(Y=1). Both or neither label may be included.
 * A singleton is not a per-example correctness probability.
 * @param {number} pHat @param {number} qHat
 * @returns {{include0:boolean,include1:boolean}}
 */
export function conformalSet(pHat, qHat) {
  probability(pHat, 'pHat');
  checkRadius(qHat);
  return { include0: pHat <= qHat, include1: 1 - pHat <= qHat };
}

/**
 * Symmetric interval using an absolute-residual threshold. Infinity is deliberate.
 * @param {number} yHat @param {number} qHat @returns {[number,number]}
 */
export function conformalInterval(yHat, qHat) {
  finite(yHat, 'yHat');
  checkRadius(qHat);
  if (qHat === Infinity) return [-Infinity, Infinity];
  const lo = yHat - qHat;
  const hi = yHat + qHat;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) throw new RangeError('interval arithmetic overflow');
  return [lo, hi];
}

/** @param {WelfordState} state */
function checkState(state) {
  if (!state || typeof state !== 'object') throw new TypeError('state must be a Welford state');
  if (!Number.isSafeInteger(state.n) || state.n < 0) throw new RangeError('state.n must be a nonnegative safe integer');
  finite(state.mean, 'state.mean');
  finite(state.m2, 'state.m2');
  if (state.m2 < 0 || (state.n <= 1 && state.m2 !== 0) || (state.n === 0 && state.mean !== 0)) {
    throw new RangeError('inconsistent Welford state');
  }
}
/** @returns {WelfordState} */
export function createWelford() { return { n: 0, mean: 0, m2: 0 }; }

/**
 * Updates the state in place. Invalid input / arithmetic overflow leaves it intact.
 * @param {WelfordState} state @param {number} x @returns {WelfordState}
 */
export function welfordUpdate(state, x) {
  checkState(state);
  finite(x, 'x');
  if (state.n === Number.MAX_SAFE_INTEGER) throw new RangeError('observation count overflow');
  const n = state.n + 1;
  const delta = x - state.mean;
  const mean = state.mean + delta / n;
  const m2 = state.m2 + delta * (x - mean);
  if (!Number.isFinite(mean) || !Number.isFinite(m2) || m2 < 0) {
    throw new RangeError('Welford arithmetic overflow or precision loss');
  }
  state.n = n;
  state.mean = mean;
  state.m2 = m2;
  return state;
}

/**
 * Default: population variance. ddof=1 requests sample variance.
 * No observations is an error, not zero variance.
 * @param {WelfordState} state @param {{ddof?:0|1}} [options]
 * @returns {{mean:number,variance:number,std:number,n:number}}
 */
export function welfordStats(state, { ddof = 0 } = {}) {
  checkState(state);
  if (ddof !== 0 && ddof !== 1) throw new RangeError('ddof must be 0 or 1');
  if (state.n <= ddof) throw new RangeError('not enough observations for requested variance');
  const variance = state.m2 / (state.n - ddof);
  return { mean: state.mean, variance, std: Math.sqrt(variance), n: state.n };
}

/**
 * Exact empirical two-sample KS statistic, including ties. Empty samples error.
 * The statistic alone does not diagnose why a distribution differs.
 * @param {NumericArray} sampleA @param {NumericArray} sampleB @returns {number}
 */
export function ksStatistic(sampleA, sampleB) {
  const a = sortedCopy(sampleA, 'sampleA');
  const b = sortedCopy(sampleB, 'sampleB');
  if (!a.length || !b.length) throw new RangeError('KS samples must not be empty');
  let i = 0; let j = 0; let D = 0;
  while (i < a.length || j < b.length) {
    const x = i < a.length && (j === b.length || a[i] <= b[j]) ? a[i] : b[j];
    while (i < a.length && a[i] <= x) i++;
    while (j < b.length && b[j] <= x) j++;
    D = Math.max(D, Math.abs(i / a.length - j / b.length));
  }
  return D;
}

/**
 * Approximate two-sided KS p-value (continuous independent samples).
 * NOT an exact finite-sample test; ties/discrete/dependent data need other methods.
 * Numerically stable complementary series for small lambda.
 * @param {number} D @param {number} n @param {number} m @returns {number}
 */
export function ksPValue(D, n, m) {
  probability(D, 'D');
  if (!Number.isSafeInteger(n) || n < 1 || !Number.isSafeInteger(m) || m < 1) {
    throw new RangeError('sample sizes must be positive safe integers');
  }
  if (D === 0) return 1;
  const en = Math.sqrt(n * m / (n + m));
  const lambda = (en + 0.12 + 0.11 / en) * D;
  if (lambda < 1.18) {
    // Jacobi theta transformation; avoids catastrophic cancellation near zero.
    if (lambda < 0.04) return 1; // Complement is below double-precision resolution.
    let cdf = 0;
    const factor = Math.sqrt(2 * Math.PI) / lambda;
    for (let k = 1; k <= 100; k++) {
      const term = factor * Math.exp(-((2 * k - 1) ** 2) * Math.PI ** 2 / (8 * lambda ** 2));
      cdf += term;
      if (term < 1e-16) break;
    }
    return Math.max(0, Math.min(1, 1 - cdf));
  }
  let survival = 0;
  for (let k = 1; k <= 100; k++) {
    const term = 2 * Math.exp(-2 * k * k * lambda * lambda);
    survival += k % 2 ? term : -term;
    if (term < 1e-16) break;
  }
  return Math.max(0, Math.min(1, survival));
}

/** @param {number} pHat @param {0|1} y @returns {number} */
export function bernoulliNonconformity(pHat, y) {
  probability(pHat, 'pHat'); binary(y, 'y');
  return y === 1 ? 1 - pHat : pHat;
}
/** Binary Brier loss for one observation. @param {number} pHat @param {0|1} y @returns {number} */
export function brierScore(pHat, y) {
  probability(pHat, 'pHat'); binary(y, 'y');
  return (pHat - y) ** 2;
}

/**
 * Equal-width, positive-class binary ECE, not top-label/multiclass ECE.
 * O(N+B), with at most 65536 bins. Empty data and mismatched arrays are errors.
 * @param {NumericArray} pHats @param {NumericArray} ys @param {number} [numBins]
 * @returns {number}
 */
export function expectedCalibrationError(pHats, ys, numBins = 10) {
  arrayType(pHats, 'pHats'); arrayType(ys, 'ys');
  if (!pHats.length || pHats.length !== ys.length) throw new RangeError('ECE arrays must be nonempty and have equal lengths');
  if (!Number.isSafeInteger(numBins) || numBins < 1 || numBins > MAX_BINS) {
    throw new RangeError(`numBins must be an integer in [1, ${MAX_BINS}]`);
  }
  const deltas = new Float64Array(numBins);
  for (let i = 0; i < pHats.length; i++) {
    const p = pHats[i]; const y = ys[i];
    // Build indexed error messages only on failure, not on every valid observation.
    if (typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1) probability(p, `pHats[${i}]`);
    if (y !== 0 && y !== 1) binary(y, `ys[${i}]`);
    let b = Math.min(numBins - 1, Math.floor(p * numBins));
    // Reconcile multiplication rounding with the documented [b/B,(b+1)/B) bins.
    if (b < numBins - 1 && p >= (b + 1) / numBins) b++;
    else if (p < b / numBins) b--;
    deltas[b] += p - y;
  }
  let total = 0;
  for (const delta of deltas) total += Math.abs(delta);
  return Math.min(1, total / pHats.length);
}

/** Finite bounds and value; reversed bounds are an error. @param {number} x @param {number} lo @param {number} hi @returns {number} */
export function clamp(x, lo, hi) {
  finite(x, 'x'); finite(lo, 'lo'); finite(hi, 'hi');
  if (lo > hi) throw new RangeError('lo must not exceed hi');
  return Math.max(lo, Math.min(hi, x));
}

export default {
  quantile, calibrate, createCalibrator, conformalSet, conformalInterval,
  createWelford, welfordUpdate, welfordStats, ksStatistic, ksPValue,
  bernoulliNonconformity, brierScore, expectedCalibrationError, clamp,
};
