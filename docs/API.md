# API contracts

## Numeric input

Arrays (including readonly arrays) and numeric typed arrays are accepted. Inputs must be dense and finite. BigInt arrays, DataView, shared-buffer arrays, strings, NaN and infinity in observations are rejected. Large inputs require memory proportional to their length; callers must impose their own upload/request limits. No external model adapter is required.

| Function | Contract | Empty input | Cost |
|---|---|---|---|
| quantile(scores,q) | q in [0,1], nearest-rank order statistic; signed finite values allowed | Error | O(n log n) time, O(n) memory |
| calibrate(scores,alpha) | Finite nonnegative scores; alpha in [0,1) | Infinity | O(n log n) time, O(n) memory |
| createCalibrator(scores) | Same score contract; immutable copied snapshot | Explicit empty-calibration status | O(n log n) setup, O(1) query, O(n) storage |
| conformalSet(pHat,qHat) | pHat in [0,1], qHat nonnegative or Infinity; returns include0/include1 | Not applicable | O(1) |
| conformalInterval(yHat,qHat) | finite prediction, nonnegative radius or Infinity; returns [lower,upper] | Not applicable | O(1) |
| createWelford() | Creates {n:0,mean:0,m2:0} | Not applicable | O(1) |
| welfordUpdate(state,x) | Mutates a valid state; finite x; numeric validation precedes mutation | First observation allowed | O(1) |
| welfordStats(state,{ddof}) | ddof 0 (population, default) or 1 (sample); requires n > ddof | Error | O(1) |
| ksStatistic(a,b) | Finite samples; handles ties in the empirical statistic | Error | Sorting dominates |
| ksPValue(D,n,m) | D in [0,1]; positive safe-integer sizes; continuous independent samples | Error | Bounded series |
| bernoulliNonconformity(pHat,y) | pHat in [0,1], y exactly 0 or 1 | Not applicable | O(1) |
| brierScore(pHat,y) | Binary squared error for one pair, not multiclass summed Brier | Not applicable | O(1) |
| expectedCalibrationError(pHats,ys,bins) | Same-length nonempty arrays; binary labels; bins integer 1..65536 | Error | O(n+B) time, O(B) memory |
| clamp(x,lo,hi) | All finite, lo <= hi | Not applicable | O(1) |

### Calibration states

`finite`: the requested rank exists.
`insufficient-data`: the requested rank exceeds the sample count; threshold is Infinity.
`empty-calibration`: no calibration observations; threshold is Infinity, no empirical evidence claimed.
`full-coverage-request`: alpha=0 requires the full output space; threshold is Infinity.

Boundary calculations use JavaScript double precision and the actual numeric alpha supplied. Results at exact decimal thresholds can be conservatively affected by floating-point rounding. Scores use an inclusive `<=` threshold, so ties are conservative. No randomized tie-breaking or conditional-coverage guarantee is implemented.

### Binary sets

A result may contain neither label, one label, or both. The library does not silently insert the most likely label into an empty set. That would change the method. Neither set size nor a KS p-value is the probability that a specific AI answer is correct.

### ECE semantics

This is positive-class binary ECE: sum over equal-width bins of bin fraction times abs(mean predicted probability minus mean label). Bins are [b/B,(b+1)/B), with 1 included in the final bin. It is not top-label confidence ECE, not multiclass ECE, and not proof that predictions are calibrated. More bins are not necessarily better.

### Errors and persistence

TypeError reports an input of the wrong type. RangeError reports an invalid domain or detected overflow. Welford numeric failures do not update its state; arbitrary proxies/accessors/frozen partial-state objects are not supported mutable states. Other functions do not mutate supplied arrays.

JavaScript JSON does not represent Infinity. Persist `qHat === Infinity` with an explicit status/tag rather than treating JSON null as zero. Metadata alone cannot establish that calibration data were honestly held out. Data leakage, selection effects, distribution shift and subgroup performance remain caller responsibilities.

### Modules

The package is ESM with explicit TypeScript declarations. There is no CommonJS export; use import or dynamic import. No Deno, Bun, Safari or Firefox certification is implied by this interface.
