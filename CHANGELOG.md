# Changelog

## 0.2.0

Breaking changes from 0.1.0:

- Removed the adaptive conformal inference functions `createACI`, `aciPredict` and `aciUpdate`.
- Invalid, empty or mismatched inputs now raise `RangeError` instead of returning misleading values.
- `calibrate` returns `Infinity` when the calibration set is too small for the requested level, instead of the largest observed score.
- Node.js 22 or newer is required.

Fixes:

- The two-sample KS p-value no longer reports a shift for nearly identical large samples.
- `ksStatistic` no longer hangs on NaN input.
- The calibration rank is selected directly, removing a floating-point off-by-one.
- Expected calibration error no longer reports 0 for empty input.

Additions:

- `createCalibrator(scores)`: sort once, query many alpha levels.
- `welfordStats(state, { ddof: 1 })` returns sample variance; the default remains population variance.
- TypeScript declarations, with an automated check that they stay compatible with the implementation.

Known limitation: the declaration verifier catches the tested drift classes but does not presently detect readonly-only drift for the inline `BinarySet` and `WelfordStats` result shapes.

## 0.1.0

Initial release.
