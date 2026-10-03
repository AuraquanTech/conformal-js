# conformal-js

Distribution-free uncertainty quantification for JavaScript and TypeScript. Zero runtime dependencies.

conformal-js provides split conformal prediction intervals and binary prediction sets, plus basic calibration diagnostics.

## What problem does it solve?

A model may predict a delivery time of 30 minutes. This library uses errors on separate, held-out examples to turn that prediction into an interval. It also supports binary prediction sets and basic model diagnostics. It does not train a model or verify whether a chatbot's sentence is true.

The runtime has no external dependencies, network requests, or Node-only imports. Give it arrays of predictions and observed outcomes.

## Install

```sh
npm install conformal-js
```

Requires Node.js 22 or newer. Other JavaScript runtimes and browsers are not yet tested; see [Known limitations](#known-limitations).

## A complete regression example

Use a fixed predictor and calibration examples not used to train, tune, or select it:

```js
import { calibrate, conformalInterval } from 'conformal-js';

// Synthetic held-out predictions and corresponding actual delivery times.
const predicted = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38];
const observed  = [21, 20, 25, 23, 30, 31, 29, 36, 35, 42];
const residuals = predicted.map((p, i) => Math.abs(observed[i] - p));
const qHat = calibrate(residuals, 0.1); // alpha is the error rate, not confidence
const [lower, upper] = conformalInterval(30, qHat);
console.log({ lower, upper }); // { lower: 26, upper: 34 }
```

The usual guarantee is **marginal coverage under exchangeability**. It averages over fresh calibration and test data. It is not a 90% correctness promise for each individual prediction, each subgroup, or shifted or time-dependent data. Wide intervals can be statistically valid but unhelpful.

## Too little data is not hidden

```js
import { createCalibrator } from 'conformal-js';
const calibration = createCalibrator([1, 2, 3, 4, 5]);
console.log(calibration.summary(0.1));
// { sampleSize: 5, alpha: 0.1, rank: 6, qHat: Infinity,
//   status: 'insufficient-data' }
```

There is no sixth calibration score. Returning the largest observed error would understate the requested uncertainty. Infinity means an unbounded interval or full label set, not a data value to feed back into calibration. `JSON.stringify` converts Infinity to null; use an explicit tagged representation when persisting it.

## Query many levels without sorting again

`createCalibrator(scores)` validates, copies and sorts once. Its `threshold(alpha)` and `summary(alpha)` query that fixed snapshot in constant time. It does not learn, adapt, mutate the input, or choose actions.

## Other tools

Binary prediction sets, Bernoulli scores, binary Brier loss, positive-class equal-width ECE, Welford mean and variance, and a two-sample KS statistic with an approximate continuous-sample p-value. Invalid inputs raise informative errors rather than being silently dropped. Plain arrays and numeric typed arrays are supported.

See [API contracts](docs/API.md), [mathematical assumptions and references](docs/MATHEMATICS.md), and the [changelog](CHANGELOG.md).

## Upgrading from 0.1.0

Version 0.2.0 is not a drop-in replacement. It removes the adaptive conformal inference (ACI) functions, rejects invalid or empty inputs with errors, and returns Infinity when the calibration set is too small for the requested level. It also fixes 0.1.0 defects, including a two-sample KS p-value that could report a shift for nearly identical large samples and a hang on NaN input. See the changelog.

## Development

```sh
npm ci --ignore-scripts
npm test
npm run lint
npm run typecheck
```

`npm run typecheck` checks the JavaScript implementation, the shipped declarations, the compile-only consumer examples, and that deliberately injected type errors are detected.

## Known limitations

- Coverage is marginal and assumes exchangeable calibration and test data. The library cannot detect time dependence, distribution shift, data leakage or subgroup effects from its inputs.
- The KS p-value is an asymptotic approximation for independent continuous samples, not an exact finite-sample test.
- Only Node.js 22 and 24 are tested. Browser, Deno and Bun use is untested.
- The declaration verifier catches the tested drift classes but does not presently detect readonly-only drift for the inline `BinarySet` and `WelfordStats` result shapes. This affects TypeScript declarations only; runtime behavior is unaffected.

## License

MIT. See [LICENSE](LICENSE).
