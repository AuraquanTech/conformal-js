import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { calibrate, quantile, createCalibrator, ksStatistic, ksPValue,
  expectedCalibrationError, createWelford, welfordUpdate, welfordStats } from '../src/conformal.js';
const ref=JSON.parse(readFileSync(new URL('fixtures/scipy-reference.json',import.meta.url),'utf8'));
const close=(a,b,tol=1e-12)=>assert.ok(Math.abs(a-b)<=tol,`${a} differs from ${b}`);
for(const [i,r] of ref.ks.entries()) test(`independent SciPy KS statistic ${i}`,()=>close(ksStatistic(r.a,r.b),r.D));
for(const [i,r] of ref.survival.entries()) test(`independent SciPy Kolmogorov survival ${i}`,()=>close(ksPValue(r.D,r.n,r.m),r.p,3e-14));
for(const [i,r] of ref.quantiles.entries()) test(`independent NumPy quantile and conformal rank ${i}`,()=>{
  assert.equal(quantile(r.a,r.q),r.quantile);
  assert.equal(calibrate(r.a,r.alpha),r.threshold==='Infinity'?Infinity:r.threshold);
});
for(const [i,r] of ref.ece.entries()) test(`independent NumPy binary ECE ${i}`,()=>close(expectedCalibrationError(r.p,r.y,r.bins),r.expected,5e-14));
for(const [i,r] of ref.welford.entries()) test(`independent NumPy streaming statistics ${i}`,()=>{
  const s=createWelford();for(const x of r.a) welfordUpdate(s,x);
  close(welfordStats(s).mean,r.mean,1e-6);
  close(welfordStats(s).variance,r.variance,1e-6);
  close(welfordStats(s,{ddof:1}).variance,r.sampleVariance,1e-6);
});
test('all possible ranks: marginal lower bound for 792 finite configurations',()=>{
  for(let n=1;n<=99;n++) for(const alpha of [0,.001,.01,.05,.1,.25,.5,.9]) {
    let covered=0;
    const population=Array.from({length:n+1},(_,i)=>i+.5);
    for(let i=0;i<population.length;i++) {
      const calibration=population.filter((_,j)=>j!==i);
      if(population[i]<=calibrate(calibration,alpha)) covered++;
    }
    assert.ok(covered/(n+1)>=1-alpha-1e-15,`n=${n} alpha=${alpha}`);
  }
});
test('ties yield conservative coverage on enumerated finite populations',()=>{
  for(const population of [[0,0,0,1],[1,1,2,2,2,3],[0,0,0,0],[1,2,2,2,2,2,7]]) {
    for(const alpha of [.1,.25,.5]) {
      const covered=population.reduce((s,x,i)=>s+Number(x<=calibrate(population.filter((_,j)=>j!==i),alpha)),0);
      assert.ok(covered/population.length>=1-alpha-1e-15);
    }
  }
});
test('sorted calibration thresholds are monotone for 500 alpha levels',()=>{
  const c=createCalibrator(Array.from({length:499},(_,i)=>(i*53)%137));
  let previous=Infinity;for(let i=0;i<500;i++){const q=c.threshold(i/500);assert.ok(q<=previous);previous=q;}
});
test('counterexample: static calibration offers no shift guarantee',()=>{
  const q=calibrate(Array.from({length:200},(_,i)=>i/200),.1);
  const shifted=Array.from({length:200},(_,i)=>2+i/200);
  assert.equal(shifted.filter(x=>x<=q).length,0);
});
test('ECE agrees at every bin boundary and adjacent float for several bin counts',()=>{
  const view=new DataView(new ArrayBuffer(8));
  const adjacent=(x,delta)=>{view.setFloat64(0,x);view.setBigUint64(0,view.getBigUint64(0)+BigInt(delta));return view.getFloat64(0);};
  for(const B of [3,7,10,100,257]) {
    const p=[];for(let b=1;b<B;b++) p.push(adjacent(b/B,-1),b/B,adjacent(b/B,1));
    const y=p.map((_,i)=>i%2);let expected=0;
    for(let b=0;b<B;b++) {
      let delta=0;for(let i=0;i<p.length;i++)if(p[i]>=b/B&&(b===B-1?p[i]<=1:p[i]<(b+1)/B))delta+=p[i]-y[i];
      expected+=Math.abs(delta)/p.length;
    }
    close(expectedCalibrationError(p,y,B),expected,1e-14);
  }
});
