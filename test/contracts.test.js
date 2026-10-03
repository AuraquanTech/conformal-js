import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import api, * as lib from '../src/conformal.js';
const { calibrate, createCalibrator, quantile, conformalSet, conformalInterval,
  createWelford, welfordUpdate, welfordStats, ksStatistic, ksPValue,
  brierScore, bernoulliNonconformity, expectedCalibrationError, clamp } = lib;
const close = (a,b,tol=1e-12) => assert.ok(Math.abs(a-b)<=tol, `${a} != ${b}`);

test('API exposes exactly the approved 14 named functions plus default', () => {
  assert.deepEqual(Object.keys(lib).filter(k=>k!=='default').sort(), Object.keys(api).sort());
  assert.equal(Object.keys(api).length,14);
  for(const name of ['createACI','aciPredict','aciUpdate']) assert.equal(name in lib,false);
});
for (const x of [NaN,Infinity,-Infinity,-0.1,1,2,'0.1',null,undefined]) {
  test(`invalid alpha rejected: ${String(x)}`, () => {
    assert.throws(()=>calibrate([1,2],x));
    assert.throws(()=>calibrate([],x));
    assert.throws(()=>createCalibrator([1]).threshold(x));
  });
}
for (const xs of [[NaN],[Infinity],[-Infinity],[-0.1],['1'],[undefined],new Array(1),new BigInt64Array(0),{},null,new DataView(new ArrayBuffer(8))]) {
  test(`calibration input rejected: ${Object.prototype.toString.call(xs)} ${String(xs)}`,()=>assert.throws(()=>calibrate(xs,0.1)));
}
for (const Ctor of [Array,Float32Array,Float64Array,Int8Array,Uint8Array,Uint8ClampedArray,Int16Array,Uint16Array,Int32Array,Uint32Array]) {
  test(`numeric container ${Ctor.name} accepted without mutation`,()=>{
    const xs=Ctor.from([4,1,3,2]); const before=Array.from(xs);
    assert.equal(quantile(xs,.5),2);
    assert.equal(calibrate(xs,.5),3);
    assert.equal(ksStatistic(xs,Ctor.from([1,2,3,4])),0);
    close(expectedCalibrationError(Ctor.from([0,1]),Ctor.from([0,1])),0);
    assert.deepEqual(Array.from(xs),before);
  });
}
test('shared typed array is rejected rather than sampled concurrently',()=>{
  assert.throws(()=>calibrate(new Float64Array(new SharedArrayBuffer(16)),.1),TypeError);
});
test('small calibration sample returns full uncertainty, not sample max',()=>{
  assert.equal(calibrate([.1,.2,.3,.4,.5],.1),Infinity);
  assert.deepEqual(conformalInterval(30,Infinity),[-Infinity,Infinity]);
  assert.deepEqual(conformalSet(.9,Infinity),{include0:true,include1:true});
});
test('empty calibration has explicit status, not an empirical success claim',()=>{
  const c=createCalibrator([]);
  assert.equal(c.threshold(.1),Infinity);
  assert.equal(c.summary(.1).status,'empty-calibration');
});
test('alpha zero always means the full set',()=>{
  assert.equal(calibrate([1,2,3],0),Infinity);
  assert.equal(createCalibrator([1]).summary(0).status,'full-coverage-request');
});
test('integer rank directly selects the n-th score at the finite boundary',()=>{
  assert.equal(calibrate([1,2,3,4,5,6,7,8,9],.1),9);
  const c=createCalibrator([1,2,3,4,5]);
  assert.equal(c.summary(.1).status,'insufficient-data');
  assert.deepEqual(c.summary(.5),{sampleSize:5,alpha:.5,rank:3,qHat:3,status:'finite'});
});
test('calibrator is immutable and snapshots the input',()=>{
  const a=[4,1,3,2]; const c=createCalibrator(a); a.fill(999);
  assert.equal(c.threshold(.5),3); assert.ok(Object.isFrozen(c));
  const s=c.summary(.5); s.qHat=999; assert.equal(c.threshold(.5),3);
  assert.throws(()=>{c.sampleSize=100;},TypeError);
});
test('calibrator alpha sweep matches direct calibration for every query',()=>{
  const xs=Array.from({length:199},(_,i)=>(i*17)%61);const c=createCalibrator(xs);
  for(let i=0;i<1000;i++) assert.equal(c.threshold(i/1000),calibrate(xs,i/1000));
});
for (const x of [NaN,Infinity,-Infinity,-.1,1.1,'0.1',null]) {
  test(`invalid binary probability rejected: ${String(x)}`,()=>{
    assert.throws(()=>conformalSet(x,.5)); assert.throws(()=>brierScore(x,0));
    assert.throws(()=>bernoulliNonconformity(x,1));
    assert.throws(()=>expectedCalibrationError([x],[0]));
  });
}
for (const y of [-1,2,.5,'1',true,null,NaN]) {
  test(`invalid label rejected: ${String(y)}`,()=>{
    assert.throws(()=>brierScore(.5,y));assert.throws(()=>bernoulliNonconformity(.5,y));
    assert.throws(()=>expectedCalibrationError([.5],[y]));
  });
}
for (const q of [NaN,-Infinity,-.1,'1',null]) {
  test(`invalid radius rejected: ${String(q)}`,()=>{
    assert.throws(()=>conformalSet(.5,q));assert.throws(()=>conformalInterval(1,q));
  });
}
test('interval rejects nonfinite center and finite arithmetic overflow',()=>{
  for(const y of [NaN,Infinity,-Infinity]) assert.throws(()=>conformalInterval(y,1));
  assert.throws(()=>conformalInterval(Number.MAX_VALUE,Number.MAX_VALUE),/overflow/);
});
test('binary set can be empty; it is not replaced by the most likely label',()=>{
  assert.deepEqual(conformalSet(.5,.1),{include0:false,include1:false});
});
test('quantile invalid q and sparse arrays error without silently sorting nonsense',()=>{
  for(const q of [NaN,-.1,1.1]) assert.throws(()=>quantile([1],q));
  assert.throws(()=>quantile(new Array(3),.5));
});
test('Welford empty, sample/population semantics',()=>{
  const s=createWelford(); assert.throws(()=>welfordStats(s));
  welfordUpdate(s,2);assert.equal(welfordStats(s).variance,0);
  assert.throws(()=>welfordStats(s,{ddof:1}));welfordUpdate(s,4);
  assert.equal(welfordStats(s).variance,1);assert.equal(welfordStats(s,{ddof:1}).variance,2);
  assert.throws(()=>welfordStats(s,{ddof:2}));
});
test('Welford failed update preserves state',()=>{
  const s=createWelford();welfordUpdate(s,Number.MAX_VALUE);const before={...s};
  for(const x of [-Number.MAX_VALUE,NaN,Infinity,'2']) {
    assert.throws(()=>welfordUpdate(s,x));assert.deepEqual(s,before);
  }
});
for (const s of [null,{}, {n:-1,mean:0,m2:0},{n:1.5,mean:0,m2:0},{n:0,mean:1,m2:0},{n:1,mean:NaN,m2:0},{n:1,mean:0,m2:-1}]) {
  test(`invalid Welford state rejected ${JSON.stringify(s)}`,()=>assert.throws(()=>welfordUpdate(s,1)));
}
test('Welford count overflow is rejected before mutation',()=>{
  const s={n:Number.MAX_SAFE_INTEGER,mean:1,m2:0};assert.throws(()=>welfordUpdate(s,1));
  assert.equal(s.n,Number.MAX_SAFE_INTEGER);
});
test('KS NaN regression is bounded in an independent process',()=>{
  const module=new URL('../src/conformal.js',import.meta.url).href;
  const source=`import {ksStatistic} from ${JSON.stringify(module)}; try {ksStatistic([NaN],[1]);process.exit(3);}catch(e){if(!(e instanceof RangeError))process.exit(4);}`;
  const p=spawnSync(process.execPath,['--input-type=module','-e',source],{timeout:1500});
  assert.equal(p.error,undefined);assert.equal(p.status,0);
});
test('KS tiny distance has p-value near one, not near zero',()=>assert.equal(ksPValue(1e-10,100,100),1));
test('KS handles ties and symmetry without modifying data',()=>{
  const a=[1,1,3],b=[1,2,2];close(ksStatistic(a,b),1/3);assert.equal(ksStatistic(a,b),ksStatistic(b,a));
  assert.deepEqual(a,[1,1,3]);assert.deepEqual(b,[1,2,2]);
});
for(const bad of [[NaN],[Infinity],[],[undefined],['1']]) {
  test(`invalid KS input ${String(bad)}`,()=>assert.throws(()=>ksStatistic([1,2],bad)));
}
test('KS sample sizes and D validated even if D=0',()=>{
  for(const n of [0,-1,.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1]) assert.throws(()=>ksPValue(0,n,1));
  for(const D of [NaN,Infinity,-.1,1.1]) assert.throws(()=>ksPValue(D,1,1));
});
test('KS p-value monotonic in D across branch boundary',()=>{
  let prev=1;for(let i=0;i<=10000;i++){const p=ksPValue(i/10000,100,200);assert.ok(p<=prev+1e-14);prev=p;}
});
test('ECE validates lengths and bounded bin allocation',()=>{
  assert.throws(()=>expectedCalibrationError([.5],[]));assert.throws(()=>expectedCalibrationError([],[1]));
  for(const b of [0,-1,.5,65537,Infinity,NaN,'10']) assert.throws(()=>expectedCalibrationError([.5],[1],b));
});
test('ECE equal-width endpoint placement',()=>{
  close(expectedCalibrationError([0,.1,.2,.29,.3,.99,1],[0,0,0,1,1,1,1],100),(.1+.2+.71+.7+.01)/7);
  close(expectedCalibrationError([.29,.2901],[0,1],100),Math.abs(.29+.2901-1)/2);
});
test('clamp validates reversed, nonfinite, and nonnumeric inputs',()=>{
  assert.throws(()=>clamp(0,1,-1));for(const x of [NaN,Infinity,'1']) assert.throws(()=>clamp(x,0,1));
  assert.throws(()=>clamp(0,0,Infinity));
});
