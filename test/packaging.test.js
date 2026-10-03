import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import vm from 'node:vm';
import { calibrate, quantile, welfordUpdate } from '../src/conformal.js';
const root=new URL('../',import.meta.url);
const source=readFileSync(new URL('src/conformal.js',root),'utf8');
const pkg=JSON.parse(readFileSync(new URL('package.json',root),'utf8'));
test('package is MIT licensed and publishable only through the verification gate',()=>{
  assert.equal(pkg.private,undefined);assert.equal(pkg.license,'MIT');
  assert.match(readFileSync(new URL('LICENSE',root),'utf8'),/^MIT License\r?\n\r?\nCopyright \(c\) 2026 Ayrton Ramos Goodman\r?\n/);
  assert.match(readFileSync(new URL('README.md',root),'utf8'),/## License\s+MIT\./);
  assert.equal(Object.keys(pkg.dependencies??{}).length,0);
  assert.equal(Object.keys(pkg.exports['.'])[0],'types');
  for(const f of pkg.files) assert.ok(readFileSync(new URL(f,root)).length>0,f);
});
test('every advertised npm script points to an included local script or installed development tool',()=>{
  for(const name of ['bench','example']){
    const path=pkg.scripts[name].split(' ')[1];assert.ok(readFileSync(new URL(path,root)).length>0,path);
  }
});
test('publication lifecycle hook runs tests, lint and every typecheck stage first',()=>{
  assert.equal(pkg.scripts.prepublishOnly,'npm test && npm run lint && '+pkg.scripts.typecheck);
  assert.equal(existsSync(new URL('scripts/block-publish.cjs',root)),false);
});
test('runtime source has no imports or Node-only IO dependency',()=>{
  assert.doesNotMatch(source,/^\s*import\s/m);
  assert.doesNotMatch(source,/\b(?:fetch|XMLHttpRequest|require|process|Buffer)\s*\(/);
  assert.doesNotMatch(source,/createACI|aciPredict|aciUpdate|windowSize|warmStartN/);
});
test('cross-realm numeric arrays supported, cross-realm DataView rejected',()=>{
  const data=vm.runInNewContext('new Float64Array([1,2,3,4])');
  assert.equal(quantile(data,.5),2);
  const view=vm.runInNewContext('new DataView(new ArrayBuffer(8))');
  assert.throws(()=>calibrate(view,.1),TypeError);
});
test('externally corrupted single-observation state is rejected',()=>{
  assert.throws(()=>welfordUpdate({n:1,mean:1,m2:2},2),RangeError);
});
test('ESM executes in a fresh JS context with no Node globals or imports',()=>{
  const driver=`import vm from 'node:vm'; import {readFileSync} from 'node:fs';
    const code=readFileSync(${JSON.stringify(new URL('src/conformal.js',root).href)}.replace('file://',''),'utf8');
    const mod=new vm.SourceTextModule(code,{context:vm.createContext({})});
    await mod.link(()=>{throw new Error('unexpected dependency');});await mod.evaluate();
    if(mod.namespace.calibrate([1,2,3,4,5],.1)!==Infinity)process.exit(2);
    if(mod.namespace.ksPValue(1e-10,100,100)!==1)process.exit(3);`;
  // Use a file URL object in the child, not platform-specific path slicing.
  const portable=driver.replace(`${JSON.stringify(new URL('src/conformal.js',root).href)}.replace('file://','')`,`new URL(${JSON.stringify(new URL('src/conformal.js',root).href)})`);
  const p=spawnSync(process.execPath,['--experimental-vm-modules','--input-type=module','-e',portable],{encoding:'utf8',timeout:2000});
  assert.equal(p.error,undefined);assert.equal(p.status,0,p.stderr);
});

// A compile-only fixture contains calls intended for static checking, not execution.
// Keep it in tsc's program, but out of node:test's automatic discovery.
test('compile-only consumer checks stay outside runtime discovery and inside tsc',()=>{
  const typePath='typecheck/consumer.mts';
  const config=JSON.parse(readFileSync(new URL('tsconfig.json',root),'utf8'));
  assert.ok(config.include.includes(typePath));
  assert.equal(existsSync(new URL('test/types.mts',root)),false);
  const fixture=readFileSync(new URL(typePath,root),'utf8');
  assert.equal((fixture.match(/@ts-expect-error/g)??[]).length,5);
  const visit=(directory)=>{
    for(const entry of readdirSync(directory,{withFileTypes:true})){
      const url=new URL(entry.name+(entry.isDirectory()?'/':''),directory);
      if(entry.isDirectory()) visit(url);
      else assert.doesNotMatch(entry.name,/\.(?:cts|mts|ts)$/,
        'runtime test directory must not contain compile-only TypeScript fixtures');
    }
  };
  visit(new URL('test/',root));
});
test('candidate and lockfile metadata identify the same version and dependencies',()=>{
  const lock=JSON.parse(readFileSync(new URL('package-lock.json',root),'utf8'));
  assert.equal(pkg.version,lock.version);
  assert.equal(pkg.version,lock.packages[''].version);
  assert.equal(pkg.name,lock.name);
  assert.deepEqual(pkg.devDependencies,lock.packages[''].devDependencies);
  assert.deepEqual(pkg.engines,lock.packages[''].engines);
});

// Include globs may prefer a neighboring declaration over its JavaScript source.
// Keep a source-only program with explicit roots and check future additions too.
test('source typecheck explicitly includes every runtime JavaScript file',()=>{
  const config=JSON.parse(readFileSync(new URL('tsconfig.source.json',root),'utf8'));
  assert.equal(config.extends,'./tsconfig.json');
  assert.deepEqual(config.include,[]);
  const actual=[];
  const visit=(directory,prefix)=>{
    for(const entry of readdirSync(directory,{withFileTypes:true})){
      const path=prefix+entry.name;
      if(entry.isDirectory()) visit(new URL(entry.name+'/',directory),path+'/');
      else if(entry.name.endsWith('.js')) actual.push(path);
    }
  };
  visit(new URL('src/',root),'src/');
  assert.ok(actual.length>0);
  assert.deepEqual([...config.files].sort(),actual.sort());
  const base=JSON.parse(readFileSync(new URL('tsconfig.json',root),'utf8'));
  for(const key of ['strict','noEmit','allowJs','checkJs']) assert.equal(base.compilerOptions[key],true);
  assert.equal(base.compilerOptions.skipLibCheck,false);
});
test('typecheck requires implementation, declaration compatibility, and error-detection controls',()=>{
  assert.equal(pkg.scripts.typecheck,
    'npm run typecheck:source && npm run typecheck:consumer && npm run typecheck:declarations && npm run typecheck:verify');
  assert.equal(pkg.scripts['typecheck:source'],'tsc -p tsconfig.source.json');
  assert.equal(pkg.scripts['typecheck:consumer'],'tsc -p tsconfig.json');
  assert.equal(pkg.scripts['typecheck:declarations'],'node scripts/verify-declarations.mjs');
  assert.equal(pkg.scripts['typecheck:verify'],'node scripts/verify-typecheck.mjs');
  assert.ok(readFileSync(new URL('scripts/verify-typecheck.mjs',root),'utf8').length>0);
  const declarationVerifier=readFileSync(new URL('scripts/verify-declarations.mjs',root),'utf8');
  assert.match(declarationVerifier,/bidirectional/);
  assert.match(declarationVerifier,/brierScore/);
  assert.match(declarationVerifier,/readonly-drift-control/);
});

test('lint gate covers all repository JavaScript rather than src only',()=>{
  assert.equal(pkg.scripts.lint,'eslint . --max-warnings=0');
  const config=readFileSync(new URL('eslint.config.js',root),'utf8');
  assert.match(config,/\*\*\/\*\.\{js,mjs,cjs\}/);
  assert.match(config,/Buffer/);
  assert.match(config,/URL/);
});

test('runtime support claim excludes end-of-life Node 18 and 20 lines',()=>{
  assert.equal(pkg.engines.node,'>=22.0.0');
});
