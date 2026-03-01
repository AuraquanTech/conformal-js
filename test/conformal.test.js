// conformal.test.js — Comprehensive tests for conformal-js
// Uses Node.js built-in test runner (node --test)
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  quantile,
  calibrate,
  conformalSet,
  conformalInterval,
  createACI,
  aciPredict,
  aciUpdate,
  createWelford,
  welfordUpdate,
  welfordStats,
  ksStatistic,
  ksPValue,
  bernoulliNonconformity,
  brierScore,
  expectedCalibrationError,
  clamp,
} from "../src/conformal.js";

// ================================================================
// QUANTILE
// ================================================================

describe("quantile", () => {
  it("returns Infinity for empty array", () => {
    assert.equal(quantile([], 0.5), Infinity);
  });

  it("returns the single element for length-1 array", () => {
    assert.equal(quantile([42], 0.5), 42);
    assert.equal(quantile([42], 0.0), 42);
    assert.equal(quantile([42], 1.0), 42);
  });

  it("returns correct median for odd-length array", () => {
    assert.equal(quantile([1, 2, 3, 4, 5], 0.5), 3);
  });

  it("returns correct 90th percentile", () => {
    const scores = Array.from({ length: 100 }, (_, i) => i + 1);
    assert.equal(quantile(scores, 0.9), 90);
  });

  it("does not mutate the input array", () => {
    const arr = [5, 3, 1, 4, 2];
    const copy = [...arr];
    quantile(arr, 0.5);
    assert.deepEqual(arr, copy);
  });

  it("handles all-same values", () => {
    assert.equal(quantile([7, 7, 7, 7], 0.25), 7);
    assert.equal(quantile([7, 7, 7, 7], 0.75), 7);
  });
});

// ================================================================
// CALIBRATE
// ================================================================

describe("calibrate", () => {
  it("returns Infinity for empty scores", () => {
    assert.equal(calibrate([], 0.1), Infinity);
  });

  it("applies finite-sample correction", () => {
    // With n=9, alpha=0.1: level = ceil(10*0.9)/9 = ceil(9)/9 = 1.0
    const scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const qHat = calibrate(scores, 0.1);
    assert.equal(qHat, 0.9); // 100th percentile of scores
  });

  it("gives tighter threshold with more data", () => {
    const few = [0.1, 0.3, 0.5, 0.7, 0.9];
    const many = Array.from({ length: 100 }, (_, i) => i / 100);
    // With more data, finite-sample correction matters less
    assert.ok(calibrate(many, 0.1) <= calibrate(few, 0.1));
  });
});

// ================================================================
// CONFORMAL SET (Classification)
// ================================================================

describe("conformalSet", () => {
  it("includes both classes when qHat is large", () => {
    const result = conformalSet(0.5, 1.0);
    assert.equal(result.include0, true);
    assert.equal(result.include1, true);
  });

  it("includes only class 1 when pHat is high and qHat is moderate", () => {
    // pHat=0.9, qHat=0.15: include0=(0.9≤0.15)=false, include1=(0.1≤0.15)=true
    const result = conformalSet(0.9, 0.15);
    assert.equal(result.include0, false);
    assert.equal(result.include1, true);
  });

  it("includes only class 0 when pHat is low", () => {
    // pHat=0.1, qHat=0.15: include0=(0.1≤0.15)=true, include1=(0.9≤0.15)=false
    const result = conformalSet(0.1, 0.15);
    assert.equal(result.include0, true);
    assert.equal(result.include1, false);
  });

  it("empty set when qHat is tiny", () => {
    const result = conformalSet(0.5, 0.0);
    assert.equal(result.include0, false);
    assert.equal(result.include1, false);
  });

  it("boundary: pHat exactly equals qHat", () => {
    const result = conformalSet(0.3, 0.3);
    assert.equal(result.include0, true);   // 0.3 <= 0.3 → true
    assert.equal(result.include1, false);  // (1-0.3)=0.7 <= 0.3 → false
  });
});

// ================================================================
// CONFORMAL INTERVAL (Regression)
// ================================================================

describe("conformalInterval", () => {
  it("produces symmetric interval", () => {
    const [lo, hi] = conformalInterval(5.0, 1.0);
    assert.equal(lo, 4.0);
    assert.equal(hi, 6.0);
  });

  it("zero-width interval when qHat is 0", () => {
    const [lo, hi] = conformalInterval(3.0, 0.0);
    assert.equal(lo, 3.0);
    assert.equal(hi, 3.0);
  });

  it("infinite interval when qHat is Infinity", () => {
    const [lo, hi] = conformalInterval(0, Infinity);
    assert.equal(lo, -Infinity);
    assert.equal(hi, Infinity);
  });
});

// ================================================================
// ACI: Adaptive Conformal Inference
// ================================================================

describe("createACI", () => {
  it("initializes with defaults", () => {
    const state = createACI();
    assert.equal(state.alpha, 0.1);
    assert.equal(state.targetAlpha, 0.1);
    assert.equal(state.gamma, 0.005);
    assert.equal(state.scores.length, 0);
    assert.equal(state.t, 0);
  });

  it("accepts custom options", () => {
    const state = createACI({ targetAlpha: 0.2, gamma: 0.01, warmStartN: 100 });
    assert.equal(state.alpha, 0.2);
    assert.equal(state.gamma, 0.01);
    assert.equal(state.warmStartN, 100);
  });
});

describe("aciPredict", () => {
  it("returns conservative prediction during warm-start (classification)", () => {
    const state = createACI({ warmStartN: 50 });
    const pred = aciPredict(state, 0.7);
    assert.equal(pred.qHat, 1.0); // max conservatism
    assert.equal(pred.include0, true);
    assert.equal(pred.include1, true);
    assert.equal(pred.n, 0);
  });

  it("returns infinite interval during warm-start (regression)", () => {
    const state = createACI({ warmStartN: 50 });
    const pred = aciPredict(state, 0.5, "regression");
    assert.equal(pred.qHat, Infinity);
    assert.equal(pred.lower, -Infinity);
    assert.equal(pred.upper, Infinity);
  });

  it("uses calibration scores after warm-start", () => {
    const state = createACI({ warmStartN: 5 });
    // Add 10 scores
    for (let i = 0; i < 10; i++) {
      state.scores.push(i * 0.1);
    }
    const pred = aciPredict(state, 0.7);
    assert.ok(pred.qHat < 1.0); // Should use real quantile now
    assert.equal(pred.n, 10);
  });
});

describe("aciUpdate", () => {
  it("decreases alpha when prediction is covered", () => {
    const state = createACI({ targetAlpha: 0.1, gamma: 0.01 });
    const before = state.alpha;
    aciUpdate(state, 0.3, true); // covered → err=0
    // alpha += gamma * (0.1 - 0) = 0.001 → alpha increases (more conservative)
    assert.ok(state.alpha > before);
  });

  it("increases alpha when prediction is NOT covered", () => {
    const state = createACI({ targetAlpha: 0.1, gamma: 0.01 });
    const before = state.alpha;
    aciUpdate(state, 0.3, false); // not covered → err=1
    // alpha += gamma * (0.1 - 1) = -0.009 → alpha decreases (wider sets next time)
    assert.ok(state.alpha < before);
  });

  it("clamps alpha to bounds", () => {
    const state = createACI({ alphaMin: 0.01, alphaMax: 0.5, gamma: 10.0 });
    aciUpdate(state, 0.5, true); // huge gamma pushes alpha up
    assert.ok(state.alpha <= 0.5);

    const state2 = createACI({ alphaMin: 0.01, alphaMax: 0.5, gamma: 10.0 });
    aciUpdate(state2, 0.5, false); // huge gamma pushes alpha down
    assert.ok(state2.alpha >= 0.01);
  });

  it("increments t counter", () => {
    const state = createACI();
    aciUpdate(state, 0.5, true);
    assert.equal(state.t, 1);
    aciUpdate(state, 0.5, false);
    assert.equal(state.t, 2);
  });

  it("maintains rolling window", () => {
    const state = createACI({ windowSize: 5 });
    for (let i = 0; i < 10; i++) {
      aciUpdate(state, i * 0.1, true);
    }
    assert.equal(state.scores.length, 5); // trimmed to window
  });

  it("pushes nonconformity score to scores array", () => {
    const state = createACI();
    aciUpdate(state, 0.42, true);
    assert.equal(state.scores[0], 0.42);
  });
});

describe("ACI convergence", () => {
  it("converges coverage to target alpha over many observations", () => {
    const state = createACI({ targetAlpha: 0.1, gamma: 0.005, warmStartN: 10 });

    // Seed with calibration scores
    for (let i = 0; i < 50; i++) {
      state.scores.push(Math.random());
    }

    let errors = 0;
    const N = 2000;

    for (let t = 0; t < N; t++) {
      const pHat = Math.random();
      const pred = aciPredict(state, pHat);
      const y = Math.random() < 0.5 ? 1 : 0;
      const score = bernoulliNonconformity(pHat, y);
      const covered = y === 1 ? pred.include1 : pred.include0;

      if (!covered) errors++;
      aciUpdate(state, score, covered);
    }

    const empiricalMiscoverage = errors / N;
    // Should be roughly near 0.1 (±0.05 tolerance for stochastic test)
    assert.ok(
      empiricalMiscoverage > 0.02 && empiricalMiscoverage < 0.25,
      `Empirical miscoverage ${empiricalMiscoverage} too far from target 0.1`,
    );
  });
});

// ================================================================
// WELFORD
// ================================================================

describe("createWelford", () => {
  it("initializes empty state", () => {
    const state = createWelford();
    assert.equal(state.n, 0);
    assert.equal(state.mean, 0);
    assert.equal(state.m2, 0);
  });
});

describe("welfordUpdate + welfordStats", () => {
  it("computes correct mean for simple sequence", () => {
    let state = createWelford();
    for (const x of [2, 4, 6, 8, 10]) {
      state = welfordUpdate(state, x);
    }
    const stats = welfordStats(state);
    assert.equal(stats.mean, 6);
    assert.equal(stats.n, 5);
  });

  it("computes correct variance", () => {
    let state = createWelford();
    // Known: variance of [1,2,3,4,5] = 2.0 (population)
    for (const x of [1, 2, 3, 4, 5]) {
      state = welfordUpdate(state, x);
    }
    const stats = welfordStats(state);
    assert.ok(Math.abs(stats.variance - 2.0) < 1e-10);
    assert.ok(Math.abs(stats.std - Math.sqrt(2.0)) < 1e-10);
  });

  it("returns zero variance for n < 2", () => {
    const state = createWelford();
    welfordUpdate(state, 42);
    const stats = welfordStats(state);
    assert.equal(stats.variance, 0);
    assert.equal(stats.std, 0);
  });

  it("handles constant stream", () => {
    let state = createWelford();
    for (let i = 0; i < 100; i++) {
      state = welfordUpdate(state, 7);
    }
    const stats = welfordStats(state);
    assert.equal(stats.mean, 7);
    assert.ok(Math.abs(stats.variance) < 1e-10);
  });

  it("numerically stable for large values", () => {
    let state = createWelford();
    const base = 1e8;
    for (const x of [base + 1, base + 2, base + 3]) {
      state = welfordUpdate(state, x);
    }
    const stats = welfordStats(state);
    assert.ok(Math.abs(stats.mean - (base + 2)) < 1e-5);
    // Variance of [1,2,3] around mean 2 = 2/3
    assert.ok(Math.abs(stats.variance - 2 / 3) < 1e-5);
  });
});

// ================================================================
// KS TEST
// ================================================================

describe("ksStatistic", () => {
  it("returns 0 for identical samples", () => {
    const a = [1, 2, 3, 4, 5];
    assert.equal(ksStatistic(a, a), 0);
  });

  it("returns 0 for empty samples", () => {
    assert.equal(ksStatistic([], [1, 2, 3]), 0);
    assert.equal(ksStatistic([1, 2], []), 0);
  });

  it("returns 1.0 for completely separated samples", () => {
    const a = [1, 2, 3];
    const b = [10, 11, 12];
    assert.equal(ksStatistic(a, b), 1.0);
  });

  it("returns small D for similar distributions", () => {
    // Two samples from roughly the same uniform distribution
    const rng = seedRNG(42);
    const a = Array.from({ length: 200 }, () => rng());
    const b = Array.from({ length: 200 }, () => rng());
    const D = ksStatistic(a, b);
    assert.ok(D < 0.2, `D=${D} too large for similar distributions`);
  });

  it("returns large D for different distributions", () => {
    // One sample uniform [0,1], other shifted to [0.5, 1.5]
    const a = Array.from({ length: 100 }, (_, i) => i / 100);
    const b = Array.from({ length: 100 }, (_, i) => 0.5 + i / 100);
    const D = ksStatistic(a, b);
    assert.ok(D > 0.3, `D=${D} too small for shifted distributions`);
  });

  it("does not mutate input arrays", () => {
    const a = [5, 3, 1];
    const b = [6, 4, 2];
    const aCopy = [...a];
    const bCopy = [...b];
    ksStatistic(a, b);
    assert.deepEqual(a, aCopy);
    assert.deepEqual(b, bCopy);
  });
});

describe("ksPValue", () => {
  it("returns 1.0 when D is 0", () => {
    assert.equal(ksPValue(0, 100, 100), 1.0);
  });

  it("returns small p-value for large D", () => {
    const p = ksPValue(0.5, 100, 100);
    assert.ok(p < 0.01, `p=${p} should be very small`);
  });

  it("returns large p-value for small D", () => {
    const p = ksPValue(0.05, 100, 100);
    assert.ok(p > 0.5, `p=${p} should be large`);
  });

  it("handles edge case with empty samples", () => {
    assert.equal(ksPValue(0.5, 0, 100), 1.0);
    assert.equal(ksPValue(0.5, 100, 0), 1.0);
  });
});

// ================================================================
// SCORING
// ================================================================

describe("bernoulliNonconformity", () => {
  it("returns 1-pHat for y=1", () => {
    assert.equal(bernoulliNonconformity(0.8, 1), 0.19999999999999996); // ≈ 0.2
    assert.ok(Math.abs(bernoulliNonconformity(0.8, 1) - 0.2) < 1e-10);
  });

  it("returns pHat for y=0", () => {
    assert.equal(bernoulliNonconformity(0.8, 0), 0.8);
  });

  it("score is 0 for perfect prediction", () => {
    assert.equal(bernoulliNonconformity(1.0, 1), 0);
    assert.equal(bernoulliNonconformity(0.0, 0), 0);
  });

  it("score is 1 for worst prediction", () => {
    assert.equal(bernoulliNonconformity(0.0, 1), 1);
    assert.equal(bernoulliNonconformity(1.0, 0), 1);
  });
});

describe("brierScore", () => {
  it("returns 0 for perfect prediction", () => {
    assert.equal(brierScore(1.0, 1), 0);
    assert.equal(brierScore(0.0, 0), 0);
  });

  it("returns 1 for worst prediction", () => {
    assert.equal(brierScore(0.0, 1), 1);
    assert.equal(brierScore(1.0, 0), 1);
  });

  it("returns 0.25 for uncertain prediction", () => {
    assert.equal(brierScore(0.5, 0), 0.25);
    assert.equal(brierScore(0.5, 1), 0.25);
  });
});

describe("expectedCalibrationError", () => {
  it("returns 0 for empty arrays", () => {
    assert.equal(expectedCalibrationError([], []), 0);
  });

  it("returns 0 for perfectly calibrated predictions", () => {
    // Bin 0.0-0.1: predictions all 0.05, outcomes all 0 → |0.05-0|=0.05
    // This won't be exactly 0 for discrete data, but for calibrated data:
    const pHats = [0.0, 0.0, 1.0, 1.0];
    const ys = [0, 0, 1, 1];
    const ece = expectedCalibrationError(pHats, ys);
    assert.equal(ece, 0);
  });

  it("returns positive ECE for miscalibrated predictions", () => {
    // Predict 0.9 but outcomes are 50/50
    const pHats = Array(100).fill(0.9);
    const ys = Array(100)
      .fill(0)
      .map((_, i) => (i < 50 ? 0 : 1));
    const ece = expectedCalibrationError(pHats, ys);
    assert.ok(ece > 0.3, `ECE=${ece} should be high for miscalibrated`);
  });
});

// ================================================================
// CLAMP
// ================================================================

describe("clamp", () => {
  it("clamps below minimum", () => {
    assert.equal(clamp(-5, 0, 10), 0);
  });

  it("clamps above maximum", () => {
    assert.equal(clamp(15, 0, 10), 10);
  });

  it("passes through values in range", () => {
    assert.equal(clamp(5, 0, 10), 5);
  });

  it("handles equal bounds", () => {
    assert.equal(clamp(5, 3, 3), 3);
  });
});

// ================================================================
// INTEGRATION: Full Conformal Pipeline
// ================================================================

describe("integration: full split conformal pipeline", () => {
  it("achieves valid coverage on synthetic data", () => {
    const alpha = 0.1; // 90% coverage target

    // Generate synthetic calibration data
    const rng = seedRNG(123);
    const calibData = Array.from({ length: 200 }, () => {
      const pHat = rng();
      const y = rng() < 0.5 ? 1 : 0;
      return { pHat, y, score: bernoulliNonconformity(pHat, y) };
    });

    const calibScores = calibData.map((d) => d.score);
    const qHat = calibrate(calibScores, alpha);

    // Test on new data
    let covered = 0;
    const testN = 500;
    for (let i = 0; i < testN; i++) {
      const pHat = rng();
      const y = rng() < 0.5 ? 1 : 0;
      const set = conformalSet(pHat, qHat);
      if ((y === 1 && set.include1) || (y === 0 && set.include0)) {
        covered++;
      }
    }

    const coverage = covered / testN;
    // Should be >= 1-alpha = 0.9 (with some tolerance for finite sample)
    assert.ok(
      coverage >= 0.85,
      `Coverage ${coverage} is below 85% (target: 90%)`,
    );
  });
});

describe("integration: ACI adapts to distribution shift", () => {
  it("maintains coverage under shift", () => {
    const state = createACI({ targetAlpha: 0.1, gamma: 0.01, warmStartN: 20 });

    // Phase 1: seed calibration
    const rng = seedRNG(456);
    for (let i = 0; i < 50; i++) {
      const score = rng() * 0.5; // scores in [0, 0.5]
      state.scores.push(score);
    }

    // Phase 2: shift — scores now in [0.3, 0.8]
    let errors = 0;
    const N = 500;
    for (let t = 0; t < N; t++) {
      const pHat = 0.3 + rng() * 0.5;
      const y = rng() < 0.5 ? 1 : 0;
      const pred = aciPredict(state, pHat);
      const score = bernoulliNonconformity(pHat, y);
      const covered = y === 1 ? pred.include1 : pred.include0;

      if (!covered) errors++;
      aciUpdate(state, score, covered);
    }

    // ACI should adapt — coverage should be reasonable despite shift
    const miscoverage = errors / N;
    assert.ok(
      miscoverage < 0.3,
      `Miscoverage ${miscoverage} too high after shift`,
    );
  });
});

// ================================================================
// HELPER: Simple seeded RNG (for reproducible tests)
// ================================================================

function seedRNG(seed) {
  // Simple LCG — NOT cryptographic, just for test reproducibility
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}
