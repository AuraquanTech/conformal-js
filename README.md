# conformal-js

Distribution-free uncertainty quantification for JavaScript.

**The first conformal prediction library for the JS/TS ecosystem.**

Zero dependencies. Works in browsers, Node.js, Deno, Bun, and service workers.

## Install

```bash
npm install conformal-js
```

## Quick Start

```js
import { calibrate, conformalSet, bernoulliNonconformity } from "conformal-js";

// 1. Compute nonconformity scores on calibration data
const calibScores = calibData.map(({ pHat, y }) =>
  bernoulliNonconformity(pHat, y)
);

// 2. Get calibration threshold (90% coverage)
const qHat = calibrate(calibScores, 0.1);

// 3. Make prediction sets on new data
const { include0, include1 } = conformalSet(newPHat, qHat);
// include0=true, include1=true → uncertain
// include0=false, include1=true → confident class 1
```

## What's Inside

| Module | Functions | Use Case |
|--------|-----------|----------|
| **Split Conformal** | `calibrate`, `conformalSet`, `conformalInterval` | Static calibration with coverage guarantees |
| **ACI** | `createACI`, `aciPredict`, `aciUpdate` | Online adaptation under distribution shift |
| **Welford** | `createWelford`, `welfordUpdate`, `welfordStats` | Streaming mean/variance (numerically stable) |
| **KS Test** | `ksStatistic`, `ksPValue` | Two-sample drift detection |
| **Scoring** | `bernoulliNonconformity`, `brierScore`, `expectedCalibrationError` | Calibration metrics |

## Adaptive Conformal Inference (ACI)

For online settings where data distribution may shift:

```js
import { createACI, aciPredict, aciUpdate, bernoulliNonconformity } from "conformal-js";

const state = createACI({ targetAlpha: 0.1, gamma: 0.005 });

// On each new observation:
const pred = aciPredict(state, pHat); // → { include0, include1, qHat, n }

// After observing true outcome:
const score = bernoulliNonconformity(pHat, trueY);
const covered = trueY === 1 ? pred.include1 : pred.include0;
aciUpdate(state, score, covered);  // α adapts automatically
```

**Guarantee**: Under exchangeability, empirical miscoverage converges to `targetAlpha`. Under distribution shift, ACI adapts via the update rule from [Gibbs & Candes (2021)](https://arxiv.org/abs/2106.00170).

## Regression Intervals

```js
import { createACI, aciPredict } from "conformal-js";

const state = createACI({ targetAlpha: 0.1 });
const pred = aciPredict(state, yHat, "regression");
// → { lower, upper, qHat, n }
```

## Drift Detection

```js
import { ksStatistic, ksPValue } from "conformal-js";

const D = ksStatistic(recentScores, priorScores);
const p = ksPValue(D, recentScores.length, priorScores.length);
if (p < 0.05) console.warn("Distribution shift detected!");
```

## Online Statistics

```js
import { createWelford, welfordUpdate, welfordStats } from "conformal-js";

let state = createWelford();
for (const x of stream) {
  state = welfordUpdate(state, x);
}
const { mean, variance, std, n } = welfordStats(state);
```

## Mathematical Foundations

### Split Conformal (Vovk et al. 2005)
```
q̂ = Quantile(scores, ceil((n+1)(1-α))/n)
C(x) = { y : s(x,y) ≤ q̂ }
Guarantee: P(Y ∈ C(X)) ≥ 1 - α
```

### ACI Update Rule (Gibbs & Candes 2021)
```
errₜ = 1{yₜ ∉ Cₜ(xₜ)}
αₜ₊₁ = clamp(αₜ + γ(α_target - errₜ), α_min, α_max)
```

### Welford Recurrence (1962)
```
δ  = xₜ - μₜ₋₁
μₜ = μₜ₋₁ + δ/nₜ
M₂ = M₂ + δ·(xₜ - μₜ)
```

### KS Test (Marsaglia Approximation)
```
D = sup|F̂_A(x) - F̂_B(x)|
λ = (√(nm/(n+m)) + 0.12 + 0.11/√(nm/(n+m))) · D
p ≈ 2·Σ (-1)^(k-1) · exp(-2k²λ²)
```

## References

1. Gibbs & Candes (2021). "[Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170)". NeurIPS.
2. Vovk, Gammerman & Shafer (2005). "Algorithmic Learning in a Random World". Springer.
3. Angelopoulos & Bates (2021). "[A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)".
4. Welford (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products". Technometrics.

## License

MIT
