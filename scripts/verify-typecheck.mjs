/**
 * Positive controls for both TypeScript programs. Run with the local development
 * compiler after npm ci. All fault injection is in memory: no disk writes/emits.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve, normalize, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const require = createRequire(import.meta.url);
const canonical = (file) => normalize(resolve(file));
const sourceFile = resolve(root, 'src/conformal.js');
const declarationFile = resolve(root, 'src/conformal.d.ts');
const consumerFile = resolve(root, 'typecheck/consumer.mts');
let ts;
try {
  // Resolve the project-local package explicitly; NODE_PATH/global packages
  // must not turn a missing locked-tool install into a false pass.
  ts = require(resolve(root, 'node_modules/typescript'));
} catch {
  console.error('Typecheck verification requires local TypeScript. Run npm ci --ignore-scripts.');
  process.exit(1);
}

const records = [];
const formatDiagnostics = (diagnostics) => ts.formatDiagnosticsWithColorAndContext(diagnostics, {
  getCanonicalFileName: canonical,
  getCurrentDirectory: () => root,
  getNewLine: () => '\n',
});

function loadConfig(name) {
  const configFile = resolve(root, name);
  const loaded = ts.readConfigFile(configFile, ts.sys.readFile);
  if (loaded.error) throw new Error(formatDiagnostics([loaded.error]));
  const parsed = ts.parseJsonConfigFileContent(loaded.config, ts.sys, root, undefined, configFile);
  if (parsed.errors.length) throw new Error(formatDiagnostics(parsed.errors));
  assert.equal(parsed.options.noEmit, true, `${name} must never emit into the source tree`);
  assert.equal(parsed.options.strict, true, `${name} must stay strict`);
  assert.equal(parsed.options.skipLibCheck, false, `${name} must check declarations`);
  return parsed;
}

function compile(config, replacement) {
  const host = ts.createCompilerHost(config.options);
  if (replacement) {
    const originalGetSourceFile = host.getSourceFile.bind(host);
    host.getSourceFile = (name, languageVersion, onError, shouldCreateNewSourceFile) => {
      if (canonical(name) === canonical(replacement.file)) {
        return ts.createSourceFile(name, replacement.text, languageVersion, true);
      }
      return originalGetSourceFile(name, languageVersion, onError, shouldCreateNewSourceFile);
    };
  }
  const program = ts.createProgram(config.fileNames, config.options, host);
  const errors = ts.getPreEmitDiagnostics(program)
    .filter((d) => d.category === ts.DiagnosticCategory.Error);
  return { program, errors };
}

function requireFile(program, file) {
  assert.ok(program.getSourceFiles().some((f) => canonical(f.fileName) === canonical(file)),
    `Compiler did not include ${relative(root, file)}`);
}

function rejects(id, config, file, text, codes, minimumStart = 0) {
  const result = compile(config, { file, text });
  requireFile(result.program, file);
  const relevant = result.errors.filter((d) => d.file &&
    canonical(d.file.fileName) === canonical(file) &&
    codes.includes(d.code) && (d.start ?? -1) >= minimumStart);
  assert.ok(relevant.length > 0,
    `${id}: expected error was not caught in ${relative(root, file)}\n${formatDiagnostics(result.errors)}`);
  records.push({ id, status: 'PASS', diagnosticCodes: [...new Set(relevant.map((d) => d.code))] });
}

try {
  const source = loadConfig('tsconfig.source.json');
  const consumer = loadConfig('tsconfig.json');
  assert.equal(source.options.allowJs, true, 'Source check must include JavaScript');
  assert.equal(source.options.checkJs, true, 'Source check must report JavaScript errors');
  assert.ok(source.fileNames.some((f) => canonical(f) === canonical(sourceFile)),
    'The JavaScript implementation must be an explicit source root');

  const implementation = compile(source);
  const declarations = compile(consumer);
  requireFile(implementation.program, sourceFile);
  requireFile(declarations.program, declarationFile);
  requireFile(declarations.program, consumerFile);
  assert.equal(implementation.errors.length, 0, formatDiagnostics(implementation.errors));
  assert.equal(declarations.errors.length, 0, formatDiagnostics(declarations.errors));
  records.push({ id: 'real-program-membership-and-clean-baselines', status: 'PASS' });

  const originalSource = readFileSync(sourceFile, 'utf8');
  rejects('implementation-error-detected', source, sourceFile,
    `${originalSource}\n/** @type {number} */\nexport const __typecheckSourceProbe = 'deliberate mismatch';\n`,
    [2322], originalSource.length);

  const originalDeclarations = readFileSync(declarationFile, 'utf8');
  rejects('declaration-error-detected', consumer, declarationFile,
    `${originalDeclarations}\nexport declare const __typecheckDeclarationProbe: __DeliberatelyMissingProbeType;\n`,
    [2304], originalDeclarations.length);

  const fixture = readFileSync(consumerFile, 'utf8');
  const directives = [...fixture.matchAll(/^\/\/ @ts-expect-error[^\r\n]*$/gm)];
  assert.equal(directives.length, 5, 'All five consumer negative checks must remain');
  for (const [index, directive] of directives.entries()) {
    const pos = directive.index;
    const changed = fixture.slice(0, pos) + '// expected diagnostic enabled for verification' +
      fixture.slice(pos + directive[0].length);
    rejects(`consumer-negative-${index + 1}-detected`, consumer, consumerFile, changed,
      [2322, 2345, 2540, 2339], pos);
  }

  console.log(JSON.stringify({
    status: 'PASS', compilerVersion: ts.version,
    sourceProgramFiles: implementation.program.getRootFileNames().map((p) => relative(root, p)),
    consumerProgramFiles: declarations.program.getRootFileNames().map((p) => relative(root, p)),
    injectedFaultsCaught: records.length - 1, mutationMode: 'memory-only', checks: records,
  }, null, 2));
} catch (error) {
  console.error(`Typecheck verification failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
