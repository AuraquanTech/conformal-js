import { performance } from 'node:perf_hooks';
import { cpus } from 'node:os';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as current from '../src/conformal.js';
const flag=process.argv.indexOf('--baseline');
const baseline=flag>=0?await import(pathToFileURL(resolve(process.argv[flag+1])).href):null;
let seed=123456;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
const scores=Array.from({length:5000},random);
const alphas=Array.from({length:200},(_,i)=>.02+.9*i/200);
const p=Array.from({length:100000},random);const y=p.map(()=>random()<.5?0:1);
let sink=0;
function measure(name,fn){
  for(let i=0;i<2;i++)sink+=fn();
  const ms=[];for(let i=0;i<7;i++){const t=performance.now();sink+=fn();ms.push(performance.now()-t);}
  const sorted=[...ms].sort((a,b)=>a-b);
  return {name,median_ms:sorted[3],min_ms:sorted[0],max_ms:sorted[6],samples_ms:ms};
}
const results=[];
if(baseline)results.push(measure('baseline repeated calibration: 5000 scores x 200 levels',()=>alphas.reduce((s,a)=>s+baseline.calibrate(scores,a),0)));
results.push(measure('candidate repeated scalar calibration: 5000 scores x 200 levels',()=>alphas.reduce((s,a)=>s+current.calibrate(scores,a),0)));
results.push(measure('candidate prepared calibration INCLUDING setup: 5000 scores x 200 levels',()=>{const c=current.createCalibrator(scores);return alphas.reduce((s,a)=>s+c.threshold(a),0);}));
for(const bins of [10,100,1000]){
  if(baseline)results.push(measure(`baseline ECE: N=100000 B=${bins}`,()=>baseline.expectedCalibrationError(p,y,bins)));
  results.push(measure(`candidate ECE: N=100000 B=${bins}`,()=>current.expectedCalibrationError(p,y,bins)));
}
console.log(JSON.stringify({node:process.version,platform:process.platform,cpu:cpus()[0]?.model,repetitions:7,warmups:2,seed:123456,notes:'Single-host microbenchmark, includes validation and copies; not an end-to-end application speedup.',results,sink},null,2));
