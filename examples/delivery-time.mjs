import { calibrate, conformalInterval, createCalibrator } from '../src/conformal.js';
const predicted = [20,22,24,26,28,30,32,34,36,38];
const observed = [21,20,25,23,30,31,29,36,35,42];
const residuals = predicted.map((p,i)=>Math.abs(observed[i]-p));
const qHat = calibrate(residuals,.1);
console.log('Synthetic delivery-time example:',conformalInterval(30,qHat));
console.log('Small-data case:', createCalibrator([1,2,3,4,5]).summary(.1));
console.log('Assumption: calibration and test examples are exchangeable and calibration was held out.');
