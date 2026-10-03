import api, {createCalibrator, calibrate, conformalInterval, conformalSet, createWelford, welfordStats, brierScore, type CalibrationSummary} from 'conformal-js';
const scores = [1,2,3] as const;
const c = createCalibrator(scores);
const summary: CalibrationSummary = c.summary(.1);
const interval: [number,number] = conformalInterval(1,calibrate(new Float64Array(scores),.1));
const set: {include0:boolean;include1:boolean} = conformalSet(.7,1);
api.welfordUpdate(createWelford(),1);
welfordStats(createWelford(),{ddof:1});
void summary; void interval; void set;
// @ts-expect-error strings are not probabilities
conformalSet('0.7',1);
// @ts-expect-error labels are binary
brierScore(.7,2);
// @ts-expect-error bigint arrays are unsupported
calibrate(new BigInt64Array(2),.1);
// @ts-expect-error snapshot metadata is immutable
c.sampleSize=2;
// @ts-expect-error excluded adaptive API must not leak into declarations
api.createACI();
